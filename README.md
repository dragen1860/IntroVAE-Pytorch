# IntroVAE-Pytorch



# HowTo
1. Download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and extract it as:
```
├── /home/i/dbs/
	├──img_align_celeba # only one folder in this directory
		├── 050939.jpg
		├── 050940.jpg
		├── 050941.jpg
		├── 050942.jpg
		├── 050943.jpg
		├── 050944.jpg
		├── 050945.jpg
```

modify `/home/i/dbs` to your specific path, making sure that the `/home/i/dbs/` comtains only ONE folder since we use
`torchvision.datasets.ImageFolder` API to load dataset. 
```python
    argparser.add_argument('--root', type=str, default='/home/i/dbs/',
                           help='root/label/*.jpg')
```

2. run `python main.py  --epoch 750000` to train from strach, and use `python main.py --resume '' --epoch 1000000` to resume training from latest checkpoint.


# Training

- training curves
![](assets/train.png)

- sampled x
![](assets/xr_750000.png)

