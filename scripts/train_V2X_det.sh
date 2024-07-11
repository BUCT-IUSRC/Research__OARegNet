python train_feats.py --batch_size 4 --epochs 100 --lr 0.001 --seed 1 --gpu 2 \
--npoints 5000 --dataset V2X --voxel_size 0.3 --ckpt_dir /data/zjy/DARegNet/ckpt/pretrained/pretrain_detector \
--use_fps --use_weights --data_list ./data/v2x_list --runname train --augment 0.5 \
--root /data/zjy/V2X