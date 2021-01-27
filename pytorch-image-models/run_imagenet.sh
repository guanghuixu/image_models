bash ./distributed_train.sh $1 $2 --model mobilenetv3_large_$3 -b $4 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064 --lr-noise 0.42 0.9

# CUDA_VISIBLE_DEVICES=6,7 bash run_imagenet.sh 2 /mnt/cephfs/dataset/imagenet 120 512
