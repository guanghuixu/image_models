## [pytorch-image-models](https://rwightman.github.io/pytorch-image-models/)
support most image models, including: 'nasnetalarge'，'mobilenetv3_large_100', 'mobilenetv3_rw','fbnetc_100','mnasnet_100','semnasnet_100','spnasnet_100'
```
import timm
model = timm.create_model('nasnetalarge')
output = model(input)
```

## DARTS
```
# cd darts/cnn
model = torch.load('darts.pth')
# model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
model.drop_path_prob = 0
output, _ = model(input)
```

## proxylessnas
```
# cd proxylessnas
from proxyless_nas import proxyless_gpu, proxyless_cifar
net = proxyless_gpu() # pretrained=True
output = net(input)
```

## SinglePathOneShot
```
# cd SinglePathOneShot/src/Evaluation/data/'(2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)'
python train.py --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
python train.py --eval --eval-resume $YOUR_WEIGHT_PATH --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
```