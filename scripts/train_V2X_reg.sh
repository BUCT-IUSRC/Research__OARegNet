python train_reg.py --batch_size 4 --epochs 100 --lr 0.001 --seed 1 --gpu 2 \
--npoints 5000 --dataset V2X --voxel_size 0.3 --ckpt_dir /data/zjy/DARegNet/ckpt/pretrained/reg/ \
--use_fps --use_weights --alpha 1.8 \
--data_list ./data/v2x_list --runname train --augment 0.0 \
--root /data/zjy/V2X --wandb_dir ./wandb --freeze_detector \
--pretrain_feats /data/zjy/DARegNet/ckpt/pretrained/pretrain_feats/V2X_ckpt_train/best_train.pth