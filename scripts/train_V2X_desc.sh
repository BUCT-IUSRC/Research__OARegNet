python train_feats_13.py --batch_size 4 --epochs 100 --lr 0.001 --seed 1 --gpu 2 \
--npoints 5000 --dataset V2X --voxel_size 0.3 --ckpt_dir /data/zjy/DARegNet/ckpt/pretrained/pretrain_feats \
--use_fps --use_weights --data_list ./data/v2x_list --runname train --augment 0.5 \
--root /data/zjy/V2X --wandb_dir ./wandb --train_desc --freeze_detector \
--pretrain_detector /data/zjy/DARegNet/ckpt/pretrained/pretrain_detector/V2X_ckpt_train/best_train.pth --use_wandb