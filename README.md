# A pytorch implementation for paper "terative Quantization: A Procrustean Approach to Learning Binary Codes for Large-scale Image Retrieval" TPAMI-2013

## REQUIREMENTS
1. pytorch 1.1
2. loguru

## DATASETS
1. [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. [Flickr25k](https://pan.baidu.com/s/1Bcr5K33l7QFwIRygNxwJ4w) Password: ve86
3. [NUS-WIDE](https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q) Password: uhr3

## USAGE
```
ITQ_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset name.
  -r ROOT, --root ROOT  Path of dataset
  -c CODE_LENGTH, --code-length CODE_LENGTH
                        Binary hash code length.(default: 12)
  -T MAX_ITER, --max-iter MAX_ITER
                        Number of iterations.(default: 50)
  -q NUM_QUERY, --num-query NUM_QUERY
                        Number of query data points.(default: 1000)
  -t NUM_TRAIN, --num-train NUM_TRAIN
                        Number of training data points.(default: 5000)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 0)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size.(default: 24)
  -a ARCH, --arch ARCH  CNN architecture.(default: vgg16)
  -k TOPK, --topk TOPK  Calculate map of top k.(default: 5000)
  -g GPU, --gpu GPU     Using gpu.(default: False)
  ```

## EXPERIMENTS
cifar10: 10000 query images, 5000 training images.


 | | 16 bits | 32 bits | 64 bits | 128 bits 
   :-:   |  :-:    |   :-:   |   :-:   |   :-:     
cifar-10 MAP@ALL| 0.1906 | 0.1944 | 0.1985 | 0.2107
