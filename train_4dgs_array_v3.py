"""
train_4dgs_array.py  (v4)

Changes from v3:
  1. Auto-compute Gaussian scale from DA3 depth + focal length
  2. Validate: len(ROI images) == da3_depth.shape[0] (English error)
  3. DA3 not found ¡æ random point cloud with warning (keeps running)
  4. Depth output ¡æ single npy: { 'depth_volume': (N_t, H, W), 'times_ms': (N_t,) }
"""

import os
import re
import glob
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from collections import deque

from gaussian_renderer import render


# ==========================================
# 1. Arguments
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./img/ball-br-closeall/")
    parser.add_argument("--out_dir", type=str, default="./output/output_4dgs")
    parser.add_argument("--da3_depth", type=str, default="./depth_all_view.npy",
                        help="DA3 depth file: (N_roi, H, W) in mm. "
                             "If not found, falls back to random init.")

    # Physical specs
    parser.add_argument("--focal_length", type=float, default=8.0, help="mm")
    parser.add_argument("--pixel_size", type=float, default=0.00185, help="mm")
    parser.add_argument("--tr", type=float, default=1 / 30320.487553, help="sec")
    parser.add_argument("--te", type=float, default=0.06, help="sec")

    # Image
    parser.add_argument("--img_w", type=int, default=160)
    parser.add_argument("--img_h", type=int, default=120)

    # Gaussians

    parser.add_argument("--z_max", type=float, default=500.0, help="Max depth for rendering (mm)")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help="Multiply DA3 depth by this to get mm. "
                             "If DA3 output is already mm, use 1.0")

    parser.add_argument("--num_points", type=int, default=-1,
                        help="Number of Gaussians. "
                             "DA3 available: -1=auto (use all valid depth points), "
                             "positive=subsample to this count. "
                             "DA3 unavailable: -1=default 5000 random, "
                             "positive=that many random points.")

    # Optimization
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=2000,
                        help="Static-only warm-up iterations (deformation OFF)")
    parser.add_argument("--lr_pos", type=float, default=0.0005)
    parser.add_argument("--lr_color", type=float, default=0.005)
    parser.add_argument("--lr_opacity", type=float, default=0.01)
    parser.add_argument("--lr_scale", type=float, default=0.003)
    parser.add_argument("--lr_rot", type=float, default=0.001)
    parser.add_argument("--lr_deform", type=float, default=0.0005)

    # Blur model
    parser.add_argument("--integration_steps", type=int, default=7)
    parser.add_argument("--views_per_step", type=int, default=4)

    # Loss
    parser.add_argument("--lambda_ssim", type=float, default=0.2)
    parser.add_argument("--lambda_deform_reg", type=float, default=0.01)
    parser.add_argument("--lambda_scale_reg", type=float, default=0.01,
                        help="Penalize large Gaussian scales. "
                             "Prevents scale explosion ¡æ forces blur from motion, not size.")
    parser.add_argument("--lambda_opacity_reg", type=float, default=0.005,
                        help="Encourage binary opacity (0 or 1). "
                             "Prevents semi-transparent fog.")
    parser.add_argument("--lambda_sharp_prior", type=float, default=0.1,
                        help="Sharp prior: penalize if single-frame render equals blur sim. "
                             "Encourages actual motion learning.")

    # Output
    parser.add_argument("--output_dt_ms", type=float, default=1.0,
                        help="Output time step in milliseconds")
    parser.add_argument("--ref_camera_idx", type=int, default=-1,
                        help="Reference camera index for output rendering. "
                             "-1 = auto (center camera)")

    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=1000,
                        help="Stop if avg loss doesn't improve for this many steps. "
                             "0 = disabled.")

    return parser.parse_args()


def compute_sharpness_scores(data_dir):
    image_files = glob.glob(os.path.join(data_dir, "images", "*.png"))
    image_files = sorted(image_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    mask_files = glob.glob(os.path.join(data_dir, "masks", "*.png"))
    mask_files = sorted(mask_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    assert len(image_files) == len(mask_files)

    scores = []
    for ii in range(len(image_files)):
        image = cv2.imread(image_files[ii])
        image = np.mean(image, -1)
        # mask = cv2.imread(mask_files[ii]) / 255.0
        # mask = mask[:, :, 0]
        # image = image * mask
        image_lp = cv2.Laplacian(image, cv2.CV_64F)
        inter_image = image_lp - (np.sum(image_lp) / np.sum(mask))
        score = np.sum(inter_image * inter_image) / np.sum(mask)
        scores.append(score)

    return np.array(scores)


def compute_sharpness_score(img_tensor):
    """
    텐서 이미지의 샤프니스 스코어 계산.
    낮을수록 샤프함 (blur measure = 1 / Laplacian variance).
    img_tensor: (3, H, W) float32 CUDA tensor
    """
    gray = img_tensor.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                               dtype=torch.float32, device=img_tensor.device).view(1, 1, 3, 3)
    lap = F.conv2d(gray, lap_kernel, padding=1)
    return 1.0 / (lap.var() + 1e-6)


# ==========================================
# 2. SSIM
# ==========================================
def ssim_loss(img1, img2, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=pad, groups=3)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=pad, groups=3)
    sigma1_sq = F.conv2d((img1 ** 2).unsqueeze(0), window, padding=pad, groups=3) - mu1 ** 2
    sigma2_sq = F.conv2d((img2 ** 2).unsqueeze(0), window, padding=pad, groups=3) - mu2 ** 2
    sigma12 = F.conv2d((img1 * img2).unsqueeze(0), window, padding=pad, groups=3) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


# ==========================================
# 3. Positional Encoding + Deformation Net
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_freqs=6):
        super().__init__()
        self.output_dim = input_dim + input_dim * num_freqs * 2
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        encoded = [x]
        for f in self.freqs:
            encoded.append(torch.sin(f * x))
            encoded.append(torch.cos(f * x))
        return torch.cat(encoded, dim=-1)


class DeformationNet(nn.Module):
    def __init__(self, pos_freqs=6, time_freqs=6, hidden=128, layers=4):
        super().__init__()
        self.pos_enc = PositionalEncoding(3, pos_freqs)
        self.time_enc = PositionalEncoding(1, time_freqs)
        in_dim = self.pos_enc.output_dim + self.time_enc.output_dim
        modules = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 2):
            modules.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        modules.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*modules)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, positions, t):
        t_vec = torch.full((positions.shape[0], 1), t, device=positions.device)
        return self.net(torch.cat([self.pos_enc(positions), self.time_enc(t_vec)], dim=-1))


# ==========================================
# 4. DA3 Depth ¡æ Point Cloud
# ==========================================
def da3_depth_to_pointcloud(depth_map, cam_tx, cam_ty, cam_tz,
                            fx, fy, img_w, img_h, depth_scale=1.0):
    """
    Single depth map ¡æ 3D point cloud in world coordinates.
    Also returns per-point scale (physical size of 1 pixel at that depth).
    """
    H, W = depth_map.shape
    depth_mm = depth_map * depth_scale

    mask = (depth_mm > 1.0) & (depth_mm < 1000.0) & np.isfinite(depth_mm)

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    cx, cy = W / 2.0, H / 2.0

    Z = depth_mm[mask]
    X_cam = (uu[mask] - cx) * Z / fx
    Y_cam = (vv[mask] - cy) * Z / fy

    X_world = X_cam + cam_tx
    Y_world = Y_cam + cam_ty
    Z_world = Z + cam_tz

    points = np.stack([X_world, Y_world, Z_world], axis=-1)

    # Per-point scale: physical size of ~1.5 pixels at this depth
    # scale = depth / focal_length_pixels * pixels_per_gaussian
    pixel_physical_size = Z / fx  # mm per pixel at this depth
    point_scales = pixel_physical_size * 1.5  # cover ~1.5 pixels

    return points.astype(np.float32), point_scales.astype(np.float32)


def load_da3_and_create_pointcloud(da3_path, cameras, fx, fy, img_w, img_h,
                                   depth_scale, num_points=-1):
    """
    Load DA3 depth ¡æ validate ¡æ unproject from multiple views ¡æ merged point cloud.
    num_points: -1=use all valid points, positive=subsample to this count.
    Returns: (points (N,3), scales (N,))
    """
    print(f"\n  Loading DA3 depth from: {da3_path}")
    da3_data = np.load(da3_path, allow_pickle=True)

    # Handle formats
    if isinstance(da3_data, np.ndarray):
        if da3_data.ndim == 3:
            n_views = da3_data.shape[0]
            print(f"  DA3 shape: {da3_data.shape} (N_roi, H, W)")
        elif da3_data.ndim == 0:
            da3_data = da3_data.item()
            n_views = len(da3_data)
            print(f"  DA3 dict with {n_views} entries")
        else:
            raise ValueError(f"Unexpected DA3 shape: {da3_data.shape}")
    else:
        raise ValueError(f"Unexpected DA3 type: {type(da3_data)}")

    # === VALIDATION: ROI count must match image count ===
    n_images = len(cameras)
    if n_views != n_images:
        raise ValueError(
            f"ROI count mismatch: DA3 depth has {n_views} views "
            f"but found {n_images} ROI images. "
            f"These must be identical. Check your data."
        )
    print(f"  Validated: {n_views} DA3 views == {n_images} ROI images ?")

    # Print depth stats
    if isinstance(da3_data, np.ndarray) and da3_data.ndim == 3:
        valid_mask = (da3_data > 0) & np.isfinite(da3_data)
        if valid_mask.any():
            d_valid = da3_data[valid_mask] * depth_scale
            print(f"  DA3 depth stats (mm): min={d_valid.min():.1f}, "
                  f"max={d_valid.max():.1f}, mean={d_valid.mean():.1f}")

    all_points = []
    all_scales = []

    # Use representative views (up to 10, evenly spaced)
    n_use = min(10, len(cameras))
    step_size = max(1, len(cameras) // n_use)
    view_indices = list(range(0, len(cameras), step_size))[:n_use]

    for vi in view_indices:
        cam = cameras[vi]
        cam_tx, cam_ty, cam_tz = cam['cam_pos']

        if isinstance(da3_data, dict):
            key = cam.get('roi_key', vi)
            if key in da3_data:
                depth_map = da3_data[key]
            elif vi < len(da3_data):
                depth_map = list(da3_data.values())[vi]
            else:
                continue
        elif isinstance(da3_data, np.ndarray) and da3_data.ndim == 3:
            depth_map = da3_data[vi]
        else:
            continue

        # Resize depth to match target image size
        if depth_map.shape != (img_h, img_w):
            depth_map = cv2.resize(depth_map.astype(np.float32), (img_w, img_h),
                                   interpolation=cv2.INTER_LINEAR)

        pts, scales = da3_depth_to_pointcloud(
            depth_map, cam_tx, cam_ty, cam_tz,
            fx, fy, img_w, img_h, depth_scale
        )
        all_points.append(pts)
        all_scales.append(scales)
        print(f"    View {vi}: {len(pts)} points from depth")

    if len(all_points) == 0:
        raise ValueError("No valid points from DA3 depth. Check depth values and depth_scale.")

    all_points = np.concatenate(all_points, axis=0)
    all_scales = np.concatenate(all_scales, axis=0)
    print(f"  Total points before subsampling: {len(all_points)}")

    # Subsample if requested
    if num_points > 0 and len(all_points) > num_points:
        indices = np.random.choice(len(all_points), num_points, replace=False)
        all_points = all_points[indices]
        all_scales = all_scales[indices]
        print(f"  Subsampled to: {len(all_points)} (requested: {num_points})")
    else:
        print(f"  Using all {len(all_points)} points (auto)")

    return all_points, all_scales


# ==========================================
# 5. 4D Gaussian Model
# ==========================================
class GaussianModel4D(nn.Module):
    def __init__(self, init_points, init_scales=None):
        """
        Args:
            init_points: (N, 3) numpy array, world coords in mm
            init_scales: (N,) numpy array, per-point physical scale in mm.
                         If None, uses default small scale.
        """
        super().__init__()
        N = init_points.shape[0]

        self.positions = nn.Parameter(torch.tensor(init_points, dtype=torch.float32))
        self.colors = nn.Parameter(torch.rand((N, 3)) * 0.5 + 0.25)  # [0.25, 0.75]
        self.opacities = nn.Parameter(torch.ones((N, 1)) * 2.0)  # sigmoid(2)?0.88

        # === AUTO SCALE from depth ===
        if init_scales is not None:
            # scales parameter is in log-space: actual_scale = exp(scales)
            # so scales = log(physical_size_mm)
            s = np.clip(init_scales, 0.01, 50.0)  # clamp to sane range
            log_scales = np.log(s)
            self.scales = nn.Parameter(
                torch.tensor(log_scales, dtype=torch.float32).unsqueeze(-1).expand(-1, 3).clone()
            )
            print(f"  Auto-scale from depth: "
                  f"min={s.min():.3f}mm, max={s.max():.3f}mm, "
                  f"mean={s.mean():.3f}mm "
                  f"(log: {log_scales.min():.2f} ~ {log_scales.max():.2f})")
        else:
            # Fallback: default scale (will likely be too small/big)
            self.scales = nn.Parameter(torch.ones((N, 3)) * -2.0)  # exp(-2)=0.135mm
            print(f"  Default scale: exp(-2) = 0.135mm")

        self.rotations = nn.Parameter(torch.zeros((N, 4)))
        self.rotations.data[:, 0] = 1.0

        self.deform_net = DeformationNet()

        print(f"  Initialized {N} Gaussians")

    def get_gaussians_at(self, t, use_deform=True):
        if use_deform:
            delta = self.deform_net(self.positions.detach(), t)
            pos_t = self.positions + delta
        else:
            pos_t = self.positions
        return pos_t, self.colors, self.opacities, self.scales, self.rotations

    def deformation_magnitude(self, t):
        return self.deform_net(self.positions.detach(), t)


# ==========================================
# 6. Data Loading
# ==========================================
def parse_camera_info(filepath, args):
    filename = os.path.basename(filepath)
    match = re.search(r'y(\d+)_x(\d+)_(\d+)', filename)
    if not match:
        raise ValueError(f"Filename format error: {filename}")

    y_val = int(match.group(1))
    x_val = int(match.group(2))
    idx = int(match.group(3))

    cam_tx = x_val * args.pixel_size
    cam_ty = y_val * args.pixel_size
    cam_tz = 0.0
    t_start = y_val * args.tr
    t_end = t_start + args.te

    img = Image.open(filepath).convert("RGB")
    img = img.resize((args.img_w, args.img_h), Image.Resampling.LANCZOS)
    img_tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()

    return img_tensor, (cam_tx, cam_ty, cam_tz), t_start, t_end, idx


def load_all_cameras(image_paths, args):
    cameras = []
    for fp in tqdm(image_paths, desc="Loading images"):
        img, cam_pos, t_start, t_end, idx = parse_camera_info(fp, args)
        cameras.append({
            'image': img,
            'cam_pos': cam_pos,
            't_start': t_start,
            't_end': t_end,
            'idx': idx,
        })
    return cameras


# ==========================================
# 7. Training (v5: coarse-to-fine + scale freeze + blur from start)
# ==========================================
def train(args, cameras, init_points, init_scales=None):
    fx = args.focal_length / args.pixel_size
    fy = fx

    has_da3 = init_scales is not None
    init_type = "DA3 depth" if has_da3 else "RANDOM"

    # Coarse-to-fine schedule: integration steps increase over training
    # Phase 1: 25% iters ¡æ 3 steps (coarse motion)
    # Phase 2: 25% iters ¡æ steps//2 (medium)
    # Phase 3: 50% iters ¡æ full steps (fine)
    p1_end = args.iters // 4
    p2_end = args.iters // 2
    max_steps = args.integration_steps

    def get_integration_steps(step):
        if step <= p1_end:
            return 3
        elif step <= p2_end:
            return max(3, max_steps // 2)
        else:
            return max_steps

    print(f"\n{'=' * 60}")
    print(f"  4DGS Array Camera Training (v5 - coarse-to-fine)")
    print(f"  Init: {init_type}")
    print(f"  Views: {len(cameras)} | Points: {len(init_points)}")
    print(f"  Focal (px): {fx:.1f}")
    print(f"  Image: {args.img_w}x{args.img_h}")
    print(f"  Coarse-to-Fine: 3¡æ{max(3, max_steps//2)}¡æ{max_steps} steps")
    print(f"    Phase 1 (step 1~{p1_end}):     3 integration steps")
    print(f"    Phase 2 (step {p1_end+1}~{p2_end}): {max(3, max_steps//2)} integration steps")
    print(f"    Phase 3 (step {p2_end+1}~{args.iters}): {max_steps} integration steps")
    print(f"  Scale: {'FROZEN (DA3 init)' if has_da3 else 'learnable'}")
    print(f"  Blur model: ON from step 1 (no static warmup)")
    print(f"  Total: {args.iters} iters")
    print(f"{'=' * 60}\n")

    model = GaussianModel4D(init_points, init_scales).cuda()

    # === Scale freeze if DA3 available ===
    # DA3 gives good initial scale ¡æ freeze to prevent bloating
    scale_lr = 0.0 if has_da3 else args.lr_scale

    optimizer = torch.optim.Adam([
        {'params': [model.positions], 'lr': args.lr_pos, 'name': 'positions'},
        {'params': [model.colors], 'lr': args.lr_color, 'name': 'colors'},
        {'params': [model.opacities], 'lr': args.lr_opacity, 'name': 'opacities'},
        {'params': [model.scales], 'lr': scale_lr, 'name': 'scales'},
        {'params': [model.rotations], 'lr': args.lr_rot, 'name': 'rotations'},
        {'params': model.deform_net.parameters(), 'lr': args.lr_deform, 'name': 'deform'},
    ])

    if has_da3:
        model.scales.requires_grad_(False)
        print(f"  Scale FROZEN at DA3 init values (not learnable)")
    else:
        print(f"  Scale learnable (lr={args.lr_scale})")

    debug_dir = os.path.join(args.out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # LR scheduler: decay to 5% by end (less aggressive than 1%)
    lr_decay_rate = (0.05) ** (1.0 / args.iters)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)

    # Scale clamp (for non-DA3 case)
    if init_scales is not None:
        init_log_scale_mean = np.log(np.clip(init_scales, 0.01, 50.0)).mean()
        SCALE_MAX = init_log_scale_mean + 1.0  # tighter: ~2.7x initial
    else:
        SCALE_MAX = 1.0
    SCALE_MIN = -6.0

    loss_history = []
    loss_window = deque(maxlen=100)
    best_loss = float('inf')
    best_state = None
    steps_without_improve = 0
    n_cams = len(cameras)
    pbar = tqdm(range(1, args.iters + 1))

    # Deformation enabled from step 1 (no warmup)
    # But deform net output starts at 0 (weights initialized to 0)
    # So first few steps are effectively static anyway

    for step in pbar: #for each iteration
        optimizer.zero_grad()

        # Coarse-to-fine integration steps
        cur_steps = get_integration_steps(step)

        # Multi-view batch
        batch_idx = np.random.choice(n_cams, min(args.views_per_step, n_cams), replace=False)

        total_loss = 0.0

        for ci in batch_idx: #each batch. each camera in (random (views_per_step) cameras)
            cam = cameras[ci]
            gt_img = cam['image']
            cam_tx, cam_ty, cam_tz = cam['cam_pos']
            t_start, t_end = cam['t_start'], cam['t_end']

            # === Always use blur model (no static-only warmup) ===
            time_steps = torch.linspace(t_start, t_end, cur_steps)
            blur_sim = 0.0
            for t in time_steps:
                pos_t, col, opa, sca, rot = model.get_gaussians_at(t.item(), use_deform=True)
                r, _, _ = render(pos_t, col, opa, sca, rot,
                                 args.img_h, args.img_w, fx, fy,
                                 cam_tx, cam_ty, cam_tz,
                                 render_depth=False, z_max=args.z_max)
                blur_sim = blur_sim + r / cur_steps

            # L1 + SSIM
            l1 = F.l1_loss(blur_sim, gt_img)
            ss = ssim_loss(blur_sim, gt_img) if args.lambda_ssim > 0 else torch.tensor(0.0)
            total_loss = total_loss + (l1 + args.lambda_ssim * ss) / len(batch_idx)

            # === Sharp prior loss ===
            # Render at exposure midpoint ¡æ should look "sharper" than GT
            # Penalize if single-frame render is as blurry as GT
            if args.lambda_sharp_prior > 0:
                t_mid = (t_start + t_end) / 2.0
                pos_mid, col_m, opa_m, sca_m, rot_m = model.get_gaussians_at(t_mid, use_deform=True)
                sharp_render, _, _ = render(pos_mid, col_m, opa_m, sca_m, rot_m,
                                            args.img_h, args.img_w, fx, fy,
                                            cam_tx, cam_ty, cam_tz,
                                            render_depth=False, z_max=args.z_max)
                # Sharp render should be sharp (low sharpness_score = more sharp)
                # Minimize sharpness_score to encourage sharpness
                sharpness_score = compute_sharpness_score(sharp_render)
                total_loss = total_loss + args.lambda_sharp_prior * sharpness_score / len(batch_idx)

        # Deformation smoothness reg
        if args.lambda_deform_reg > 0:
            t_rand = np.random.uniform(cameras[0]['t_start'], cameras[-1]['t_end'])
            d1 = model.deformation_magnitude(t_rand)
            d2 = model.deformation_magnitude(t_rand + 0.001)
            reg = ((d2 - d1) ** 2).mean()
            total_loss = total_loss + args.lambda_deform_reg * reg

        # Scale regularization (only if scale is learnable)
        if not has_da3 and args.lambda_scale_reg > 0:
            scale_penalty = torch.exp(model.scales).mean()
            total_loss = total_loss + args.lambda_scale_reg * scale_penalty

        # Opacity regularization
        if args.lambda_opacity_reg > 0:
            opa = torch.sigmoid(model.opacities)
            opa_clamped = opa.clamp(1e-4, 1 - 1e-4)
            entropy = -(opa_clamped * torch.log(opa_clamped) +
                        (1 - opa_clamped) * torch.log(1 - opa_clamped))
            total_loss = total_loss + args.lambda_opacity_reg * entropy.mean()

        # NaN guard
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n  [WARNING] NaN/Inf loss at step {step}, skipping.")
            optimizer.zero_grad()
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Scale clamp (for non-DA3 case)
        if not has_da3:
            with torch.no_grad():
                model.scales.data.clamp_(SCALE_MIN, SCALE_MAX)

        lv = total_loss.item()
        loss_history.append(lv)
        loss_window.append(lv)
        avg_loss = np.mean(loss_window)

        # Track best model + early stopping
        if avg_loss < best_loss and step > 100:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            steps_without_improve = 0
        else:
            steps_without_improve += 1

        if args.early_stop_patience > 0 and steps_without_improve >= args.early_stop_patience and step > 500:
            print(f"\n  [EARLY STOP] No improvement for {args.early_stop_patience} steps "
                  f"(best avg={best_loss:.4f}, current avg={avg_loss:.4f})")
            break

        current_lr = scheduler.get_last_lr()[0]
        phase = f"C2F:{cur_steps}steps"
        pbar.set_description(f"[{phase}] L={lv:.4f} avg={avg_loss:.4f} lr={current_lr:.6f}")

        # Debug vis
        if step % args.save_every == 0:
            _save_debug(model, cameras[0], fx, fy, args, debug_dir, step, True)

    # Restore best model if training diverged
    if best_state is not None and avg_loss > best_loss * 1.5:
        print(f"\n  [INFO] Final loss ({avg_loss:.4f}) > 1.5x best ({best_loss:.4f})")
        print(f"  Restoring best checkpoint.")
        model.load_state_dict({k: v.cuda() for k, v in best_state.items()})

    np.save(os.path.join(args.out_dir, "loss_history.npy"), np.array(loss_history))
    return model, fx, fy


# ==========================================
# 8. Time-Based Output
# ==========================================
def extract_temporal_results(args, model, cameras, fx, fy):
    """
    Time-axis output:
      - sharp_frames/  ¡æ per-time PNG images
      - depth_volume.npz ¡æ { 'depth_mm': (N_t, H, W), 'times_ms': (N_t,) }
      - depth_color/   ¡æ per-time colormapped PNG
    """
    t_min = min(c['t_start'] for c in cameras)
    t_max = max(c['t_end'] for c in cameras)
    dt = args.output_dt_ms / 1000.0

    time_points = np.arange(t_min, t_max, dt)
    times_ms = time_points * 1000.0

    print(f"\n  Time range: {t_min*1000:.2f}ms ~ {t_max*1000:.2f}ms")
    print(f"  Output dt: {args.output_dt_ms}ms ¡æ {len(time_points)} frames")

    # Reference camera
    if args.ref_camera_idx >= 0:
        ref_cam = cameras[args.ref_camera_idx]
    else:
        ref_cam = cameras[len(cameras) // 2]

    ref_tx, ref_ty, ref_tz = ref_cam['cam_pos']
    print(f"  Reference camera: pos=({ref_tx:.3f}, {ref_ty:.3f}, {ref_tz:.3f})mm")

    # Output directories
    sharp_dir = os.path.join(args.out_dir, "sharp_frames")
    depth_color_dir = os.path.join(args.out_dir, "depth_color")
    os.makedirs(sharp_dir, exist_ok=True)
    os.makedirs(depth_color_dir, exist_ok=True)

    # Pre-scan depth range for consistent colormap
    print("  Computing depth range...")
    all_depths = []
    with torch.no_grad():
        sample_step = max(1, len(time_points) // 20)
        for t_sec in time_points[::sample_step]:
            pos_t, col, opa, sca, rot = model.get_gaussians_at(t_sec, use_deform=True)
            _, d, _ = render(pos_t, col, opa, sca, rot,
                             args.img_h, args.img_w, fx, fy,
                             ref_tx, ref_ty, ref_tz,
                             render_depth=True, z_max=args.z_max)
            valid = d[d > 0]
            if len(valid) > 0:
                all_depths.extend([valid.min().item(), valid.max().item()])

    if all_depths:
        d_global_min = min(all_depths)
        d_global_max = max(all_depths)
    else:
        d_global_min, d_global_max = 0, args.z_max
    print(f"  Depth range: {d_global_min:.1f} ~ {d_global_max:.1f} mm")

    # === Render all time steps ===
    # Collect depth volume in memory
    depth_volume = np.zeros((len(time_points), args.img_h, args.img_w), dtype=np.float32)

    print(f"\n  Rendering {len(time_points)} frames...")
    with torch.no_grad():
        for i, t_sec in enumerate(tqdm(time_points, desc="Rendering")):
            t_ms = t_sec * 1000.0

            pos_t, col, opa, sca, rot = model.get_gaussians_at(t_sec, use_deform=True)
            sharp_img, depth_mm, _ = render(
                pos_t, col, opa, sca, rot,
                args.img_h, args.img_w, fx, fy,
                ref_tx, ref_ty, ref_tz,
                render_depth=True, z_max=args.z_max,
            )

            # Sharp image ¡æ PNG
            img_np = (sharp_img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(sharp_dir, f"t_{t_ms:08.3f}ms.png"),
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            )

            # Depth ¡æ volume array
            d_np = depth_mm[0].cpu().numpy()
            depth_volume[i] = d_np

            # Depth colormap ¡æ PNG
            d_range = max(d_global_max - d_global_min, 1e-4)
            d_norm = np.clip((d_np - d_global_min) / d_range, 0, 1)
            d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.putText(d_color, f"t={t_ms:.1f}ms",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(d_color, f"{d_global_min:.0f}-{d_global_max:.0f}mm",
                        (4, args.img_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.imwrite(
                os.path.join(depth_color_dir, f"t_{t_ms:08.3f}ms.png"),
                d_color
            )

    # === Save single depth volume NPZ ===
    depth_out_path = os.path.join(args.out_dir, "depth_volume.npz")
    np.savez(depth_out_path,
             depth_mm=depth_volume,      # (N_t, H, W) float32, mm
             times_ms=times_ms.astype(np.float32))  # (N_t,) float32, ms
    print(f"\n  Depth volume saved: {depth_out_path}")
    print(f"    depth_mm shape: {depth_volume.shape}  (N_time, H, W)")
    print(f"    times_ms shape: {times_ms.shape}  (N_time,)")

    print(f"\n  Output saved:")
    print(f"    Sharp frames:  {sharp_dir}/")
    print(f"    Depth volume:  {depth_out_path}")
    print(f"    Depth color:   {depth_color_dir}/")


# ==========================================
# 9. Debug Visualization Helper
# ==========================================
def _save_debug(model, cam, fx, fy, args, debug_dir, step, use_deform):
    with torch.no_grad():
        gt_img = cam['image']
        cam_tx, cam_ty, cam_tz = cam['cam_pos']
        t_start, t_end = cam['t_start'], cam['t_end']

        # Sim blur
        time_steps = torch.linspace(t_start, t_end, args.integration_steps)
        blur_vis = 0.0
        for t in time_steps:
            pos_t, col, opa, sca, rot = model.get_gaussians_at(t.item(), use_deform=use_deform)
            r, _, _ = render(pos_t, col, opa, sca, rot,
                             args.img_h, args.img_w, fx, fy,
                             cam_tx, cam_ty, cam_tz,
                             render_depth=False, z_max=args.z_max)
            blur_vis = blur_vis + r / args.integration_steps

        # Sharp at mid
        t_mid = (t_start + t_end) / 2.0
        pos_m, col_m, opa_m, sca_m, rot_m = model.get_gaussians_at(t_mid, use_deform=use_deform)
        sharp_vis, depth_vis, _ = render(
            pos_m, col_m, opa_m, sca_m, rot_m,
            args.img_h, args.img_w, fx, fy,
            cam_tx, cam_ty, cam_tz,
            render_depth=True, z_max=args.z_max,
        )

        gt_np = _to_cv2(gt_img)
        sim_np = _to_cv2(blur_vis)
        sharp_np = _to_cv2(sharp_vis)

        d_np = depth_vis[0].cpu().numpy()
        d_valid = d_np[d_np > 0]
        if len(d_valid) > 0:
            d_norm = np.clip((d_np - d_valid.min()) / max(d_valid.max() - d_valid.min(), 1e-4), 0, 1)
            d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.putText(d_color, f"{d_valid.min():.0f}-{d_valid.max():.0f}mm",
                        (4, args.img_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        else:
            d_color = np.zeros((args.img_h, args.img_w, 3), dtype=np.uint8)

        phase = "STATIC" if not use_deform else "4D"
        cv2.putText(gt_np, "GT Blur", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(sim_np, f"Sim [{phase}]", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(sharp_np, "Sharp", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(d_color, "Depth(mm)", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        combined = np.concatenate([gt_np, sim_np, sharp_np, d_color], axis=1)
        cv2.imwrite(os.path.join(debug_dir, f"step_{step:05d}.png"), combined)


def _to_cv2(tensor_3hw):
    np_img = (tensor_3hw.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    image_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.tif")))

    if not image_paths:
        print(f"[Error] No .tif files in {args.data_dir}")
        exit(1)

    print(f"  Found {len(image_paths)} images")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load cameras
    cameras = load_all_cameras(image_paths, args)

    fx = args.focal_length / args.pixel_size
    fy = fx

    # ============================================
    # DA3 Depth ¡æ Point Cloud ¡æ Gaussian Init
    # ============================================
    init_scales = None

    if args.da3_depth and os.path.exists(args.da3_depth):
        init_points, init_scales = load_da3_and_create_pointcloud(
            args.da3_depth, cameras, fx, fy,
            args.img_w, args.img_h,
            args.depth_scale, args.num_points
        )
    else:
        if args.da3_depth and not os.path.exists(args.da3_depth):
            print(f"\n  [WARNING] DA3 depth file not found: {args.da3_depth}")
        else:
            print(f"\n  [WARNING] No DA3 depth path specified.")

        N = args.num_points if args.num_points > 0 else 5000
        print(f"  Creating RANDOM point cloud ({N} points).")
        print(f"  Convergence will be very slow without DA3 initialization.\n")

        init_points = np.zeros((N, 3), dtype=np.float32)
        init_points[:, 0] = (np.random.rand(N) - 0.5) * 50.0
        init_points[:, 1] = (np.random.rand(N) - 0.5) * 50.0
        init_points[:, 2] = np.random.rand(N) * 200.0 + 100.0  # 100~300mm
        init_scales = None  # will use default scale

    # Train
    model, fx, fy = train(args, cameras, init_points, init_scales)

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'args': vars(args),
    }, os.path.join(args.out_dir, "model.pth"))

    # ============================================
    # Time-based output
    # ============================================
    extract_temporal_results(args, model, cameras, fx, fy)

    print(f"\n  All done.")