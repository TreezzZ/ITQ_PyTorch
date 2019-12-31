# A pytorch implementation for paper "Iterative Quantization: A Procrustean Approach to Learning Binary Codes for Large-scale Image Retrieval" TPAMI-2013

## REQUIREMENTS
`pip install -r requirements.txt`

1. pytorch >= 1.0
2. loguru

## DATASETS
1. [cifar10-gist.mat](https://pan.baidu.com/s/1qE9KiAOTNs5ORn_WoDDwUg) password: umb6
2. [cifar-10_alexnet.t](https://pan.baidu.com/s/1ciJIYGCfS3m0marQvatNjQ) password: f1b7
3. [nus-wide-tc21_alexnet.t](https://pan.baidu.com/s/1YglFwoxB-3j7xTEyAc8ykw) password: vfeu
4. [imagenet-tc100_alexnet.t](https://pan.baidu.com/s/1ayv4wdtCOzEDsJy01SjRew) password: 6w5i

## USAGE
```
usage: run.py [-h] [--dataset DATASET] [--root ROOT]
              [--code-length CODE_LENGTH] [--max-iter MAX_ITER] [--topk TOPK]
              [--gpu GPU]

ITQ_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --code-length CODE_LENGTH
                        Binary hash code length.(default:
                        8,16,24,32,48,64,96,128)
  --max-iter MAX_ITER   Number of iterations.(default: 3)
  --topk TOPK           Calculate map of top k.(default: ALL)
  --gpu GPU             Using gpu.(default: False)
```

## EXPERIMENTS
1. cifar10-gist dataset. Gist features, 1000 query images, 5000 training images.
2. cifar-10-alexnet dataset. Alexnet features, 1000 query images, 5000 training images.
3. nus-wide-tc21-alexnet dataset. Alexnet features, top 21 classes, 2100 query images, 10500 training images.
4. imagenet-tc100-alexnet dataset. Alexnet features, top 100 classes, 5000 query images, 10000 training images.

   Bits     | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 
   ---        |   ---  |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   
  cifar10-gist@ALL  | 0.1457 | 0.1603 | 0.1613 | 0.1654 | 0.1702 | 0.1713 | 0.1748 | 0.1762
  cifar10-alexnet@ALL | 0.1985 | 0.2176 | 0.2210 | 0.2409 | 0.2471 | 0.2453 | 0.2582 | 0.2580
  nus-wide-tc21-alexnet@5000 | 0.6540 | 0.6913 | 0.7030 | 0.7171 | 0.7314 | 0.7382 | 0.7488 | 0.7554
  imagenet-tc100-alexnet@1000 | 0.1552 | 0.2284 | 0.2808 | 0.3240 | 0.3743 | 0.4090 | 0.4449 | 0.4575

   Bits     | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 
   ---        |   ---  |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   
  cifar10-gist@ALL  | 0.1457 | 0.1603  | 0.1613  | 0.1654  | 0.1702  | 0.1713  | 0.1748  | 0.1762
  cifar10-alexnet@ALL | 0.1985 | 0.2175 | 0.2210 | 0.2409 | 0.2471 | 0.2453 | 0.2582 | 0.2580
  nus-wide-tc21-alexnet@5000 | 0.6540 | 0.6913 | 0.7030 | 0.7171 | 0.7314 | 0.7382 | 0.7488 | 0.7554
  imagenet-tc100-alexnet@1000 | 0.1552 | 0.2284 | 0.2808 | 0.3240 | 0.3743 | 0.4090 | 0.4449 | 0.4575

