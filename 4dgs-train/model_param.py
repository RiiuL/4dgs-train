import torch

ckpt = torch.load("output/output_4dgs_/model.pth")

# ¾î¶² ÆÄ¶ó¹ÌÅÍ·Î µ¹·È´ÂÁö
print(ckpt['args'])
# {'focal_length': 0.6, 'te': 0.06, 'iters': 10000, 'num_points': -1, ...}

# ÇÐ½ÀµÈ Gaussianµé
print(ckpt['model_state'].keys())
# positions, colors, opacities, scales, rotations, deform_net.*