"""
gaussian_renderer.py
diff-gaussian-rasterization wrapper for MLA array camera.
Metric depth output in mm.
"""

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def make_view_matrix(cam_tx, cam_ty, cam_tz):
    W = torch.eye(4, dtype=torch.float32)
    W[0, 3] = -cam_tx
    W[1, 3] = -cam_ty
    W[2, 3] = -cam_tz
    return W


def make_proj_matrix(fx, fy, img_w, img_h, znear=0.01, zfar=1000.0):
    P = torch.zeros((4, 4), dtype=torch.float32)
    P[0, 0] = 2.0 * fx / img_w
    P[1, 1] = 2.0 * fy / img_h
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[3, 2] = -1.0
    return P


def render(positions, colors, opacities, scales, rotations,
           img_h, img_w, fx, fy,
           cam_tx, cam_ty, cam_tz,
           bg_color=None, scale_modifier=1.0,
           render_depth=True, z_max=500.0):
    device = positions.device
    N = positions.shape[0]

    if bg_color is None:
        bg_color = torch.zeros(3, device=device)
    else:
        bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)

    view_matrix = make_view_matrix(cam_tx, cam_ty, cam_tz).to(device)
    proj_matrix = make_proj_matrix(fx, fy, img_w, img_h).to(device)
    full_proj = proj_matrix @ view_matrix

    tanfovx = img_w / (2.0 * fx)
    tanfovy = img_h / (2.0 * fy)
    campos = torch.tensor([cam_tx, cam_ty, cam_tz], dtype=torch.float32, device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=img_h, image_width=img_w,
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg_color, scale_modifier=scale_modifier,
        viewmatrix=view_matrix.T.contiguous(),
        projmatrix=full_proj.T.contiguous(),
        sh_degree=0, campos=campos,
        prefiltered=False, debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros((N, 3), dtype=torch.float32, device=device, requires_grad=True)

    colors_act = torch.sigmoid(colors)
    opacities_act = torch.sigmoid(opacities)
    scales_act = torch.exp(scales)
    rotations_norm = torch.nn.functional.normalize(rotations, dim=-1)

    rendered_image, radii = rasterizer(
        means3D=positions, means2D=means2D,
        shs=None, colors_precomp=colors_act,
        opacities=opacities_act, scales=scales_act,
        rotations=rotations_norm, cov3D_precomp=None,
    )

    depth_map = None
    if render_depth:
        with torch.no_grad():
            pos_cam = positions - campos.unsqueeze(0)
            z_vals = pos_cam[:, 2:3].clamp(min=0.01)
            z_norm = (z_vals / z_max).clamp(0.0, 1.0)
            z_rgb = z_norm.expand(-1, 3)

        depth_settings = GaussianRasterizationSettings(
            image_height=img_h, image_width=img_w,
            tanfovx=tanfovx, tanfovy=tanfovy,
            bg=torch.zeros(3, device=device),
            scale_modifier=scale_modifier,
            viewmatrix=view_matrix.T.contiguous(),
            projmatrix=full_proj.T.contiguous(),
            sh_degree=0, campos=campos,
            prefiltered=False, debug=False,
        )
        depth_rast = GaussianRasterizer(depth_settings)
        depth_rgb, _ = depth_rast(
            means3D=positions,
            means2D=torch.zeros_like(means2D),
            shs=None, colors_precomp=z_rgb,
            opacities=opacities_act, scales=scales_act,
            rotations=rotations_norm, cov3D_precomp=None,
        )
        depth_map = depth_rgb[0:1] * z_max  # (1, H, W) mm

    return rendered_image, depth_map, radii