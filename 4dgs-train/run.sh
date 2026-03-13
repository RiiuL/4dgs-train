#python train_4dgs_array.py \
#  --data_dir "../imgall/PatterendCutball-close.jpg" \
#  --da3_depth "../Depth-Anything-3/img/PatterendCutball-close.jpg/da3_results/npy/depth_all_views.npy" \
#  --focal_length 0.6 \
#  --pixel_size 0.00185 \
#  --te 0.06 \
#  --img_w 80 \
#  --img_h 60 \
#  --max_points 500 \
#  --iters 10000 \
#  --integration_steps 3 \
#  --views_per_step 4 \
#  --lambda_ssim 0.2 \
#  --lambda_deform_reg 0.01 \
#  --depth_scale 1.0
#change test
#  --da3_depth "../Depth-Anything-3/img/PatterendCutball-close.jpg/da3_results/npy/depth_all_views.npy" \
python train_4dgs_array_v3.py \
  --data_dir "../imgall/PatterendCutball-close.jpg" \
  --focal_length 0.6 \
  --pixel_size 0.00185 \
  --te 0.06 \
  --img_w 80 \
  --img_h 60 \
  --num_points 500 \
  --iters 5000 \
  --integration_steps 5 \
  --views_per_step 5 \
  --lambda_ssim 0.2 \
  --lambda_deform_reg 0.1 \
  --lambda_scale_reg 0.05 \
  --lambda_opacity_reg 0.01 \
  --lr_scale 0.005 \
  --lr_deform 0.001 \
  --warmup_iters 20 \
  --depth_scale 1.0
  #focal length, pixel size mm
  # te : sec
  # tr : sec
