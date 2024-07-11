python test.py --seed 1 --gpu 2 --npoints 5000 --dataset V2X --voxel_size 0.3 \
--use_fps --use_weights --data_list ./data/v2x_list --root /data/zjy/V2X \
--pretrain_weights /data/zjy/DARegNet/ckpt/pretrained/reg/V2X_ckpt_train/best_train.pth \
--save_dir ./save/V2X_ckpt_train/2.5_5.0