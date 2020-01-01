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
cifar10-gist dataset. Gist features, 1000 query images, 5000 training images, MAP@ALL.

cifar-10-alexnet dataset. Alexnet features, 1000 query images, 5000 training images, MAP@ALL.

nus-wide-tc21-alexnet dataset. Alexnet features, top 21 classes, 2100 query images, 10500 training images, MAP@5000.

imagenet-tc100-alexnet dataset. Alexnet features, top 100 classes, 5000 query images, 10000 training images, MAP@1000.

   Bits     | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 
   ---        |   ---  |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   
  cifar10-gist@ALL  | 0.1484 | 0.1584 | 0.1613 | 0.1632 | 0.1672 | 0.1688 | 0.1726  | 0.1749
  cifar10-alexnet@ALL | 0.2000 | 0.2175 | 0.2215 | 0.2308 | 0.2386 | 0.2490 | 0.2551 | 0.2623
  nus-wide-tc21-alexnet@5000 | 0.6423 | 0.6878 | 0.7016 | 0.7186 | 0.7280 | 0.7389 | 0.7500 | 0.7539
  imagenet-tc100-alexnet@1000 | 0.1617 | 0.2369 | 0.2732 | 0.3296 | 0.3751 | 0.4076 | 0.4418 | 0.4554

