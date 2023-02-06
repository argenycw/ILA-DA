# ILA-DA
This is the code for the Paper submitted to ICLR2023: 

[ILA-DA: Improving Transferability of Intermediate Level Attack with Data Augmentation](https://openreview.net/forum?id=OM7doLjQbOQ)


## Environments
The experiments in the paper are conducted under the following environment:
* Python 3.8.5
* Numpy 1.19.2
* Pillow 8.0.0
* PyTorch 1.9.0
* Torchvision 0.10.0

## Datasets

### ImageNet
The ImageNet validation dataset should be prepared into the following structure:
```
ila-da/
└── data/
    ├── selected_imagenet_full.csv
    └── ILSVRC2012_img_val/
        ├── n01440764/
        │   ├── ILSVRC2012_val_00000293.JPEG
        │   └── ...
        ├── n01443537/
        │   ├── ILSVRC2012_val_00000236.JPEG
        │   └── ...
        ...
        .            
        └── n15075141/
```
`ILSVRC2012_img_val` is a folder containing 1000 subfolders, corresponding to the 1000 ImageNet classes. Each subfolder contains all valiation images belonging to that class.

`selected_imagenet_full.csv` should be in the following format:
```
n01440764,ILSVRC2012_val_00045866.JPEG,...
n01443537,ILSVRC2012_val_00013623.JPEG,...
...
```
The first column is the name of the folders (can be in any format as long as it matches with the folder names), the remaining columns are the names of all selected images inside. It is fine for the rows to have different number of separated values.

### CIFAR10 & CIFAR100

The CIFAR10 and CIFAR100 datasets and pretrained models are recommended to be arranged in the following format.
```
ila-da/
├── bearpaw_models/
│   ├── __init__.py
│   └── cifar/
└── data/
    ├── cifar10/
    │   ├── vgg19_bn/
    │   └── ...
    ├── cifar100/
    │   ├── vgg19_bn/
    │   └── ...
    ├── cifar10_batches.pt
    ├── cifar10_labels.pt
    ├── cifar100_batches.pt        
    └── cifar100_labels.pt
```
We adopt Bearpaw for the [model definitions](https://github.com/bearpaw/pytorch-classification/tree/master/models) and [pretrained model weights](https://mycuhk-my.sharepoint.com/personal/1155056070_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155056070%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Frelease%2Fpytorch%2Dclassification%2Fcheckpoints&ga=1). The `.pt` files for CIFAR10 and CIFAR100 can be downloaded [here]().

To run attack on CIFAR10 and CIFAR100, run `attack_cifar.py` instead. Most of the commands are just the same as `attack.py`.

## Attacking Undefended Models
```
usage: attack.py [-h] [--dataroot DATAROOT] [--gpu GPU] [--batch_size BATCH_SIZE] [--epsilon EPSILON] [--max_batch MAX_BATCH] [--save_dir SAVE_DIR] [-m MODEL_TYPE]
                 [--niters NITERS] [--ila ILA] [--ila_layer ILA_LAYER] [--step_size STEP_SIZE] [--pgd] [-a ALPHA] [--aug AUG] [--search SEARCH] [--eta ETA]

For Generation of Transferrable Attack

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   location of the data
  --gpu GPU             gpu device id
  --batch_size BATCH_SIZE
  --epsilon EPSILON     The perturbation budget
  --max_batch MAX_BATCH
                        The number of batch to stop early (default: -1 = not to stop)
  --save_dir SAVE_DIR   The path to save the output tensors for validation
  -m MODEL_TYPE, --model_type MODEL_TYPE
                        resnet50/vgg19/inception_v3
  --niters NITERS       The number of attack iteration
  --ila ILA             The number of ILA iteration after the attack generation
  --ila_layer ILA_LAYER
                        The layer to perform ILA (note the format is different for different model)
  --step_size STEP_SIZE
                        The step size of both I-FGSM and ILA
  --pgd                 to activate random initialization (PGD)
  -a ALPHA, --alpha ALPHA
                        The alpha value for attack interpolation (< 0 for being adaptive)
  --aug AUG             t/c/r/j/a (traslation/cropping/rotation/color jitter/adversarial)
  --search SEARCH       to activate augmentation searach
  --eta ETA             The step size of augmentation update
```

To generate an I-FGSM (10 iters) + ILA-DA (50 iters) attack at layer '3-1' of ResNet50 with $\epsilon = 0.03$:
```
python attack.py
```
The command above is equivalent to:
```
python attack.py --search 'gumbel' --eta 5e-4 --epsilon 0.03 --batch_size 100 --niters 10 --ila 50 -m resnet50 --ila_layer 3_1 --aug atrc --alpha -1
```
By default, the adversarial examples will be saved into `data/attack_batches/bs100_eps0.03_resnet50_niters10_ila50_l3_1_alpha-1_augatrc_searchgumbel_eta5e-05` (folder name will depend on input arguments) as batched PyTorch tensors.

To perform a standard ILA without any augmentation at layer 9 of VGG19 with $\epsilon = 0.05$, run
```
python attack.py --epsilon 0.05 -m vgg19 --ila 0 --aug _ --alpha 0 --search none
```

## Attacking Defended Models
To attack defended models, run `attack_def.py` instead of `attack.py`. `attack_def.py` is mostly the same as `attack.py` except that:
- ILA-DA is run directly without running a reference attack. Hence, the path of the reference attack (in PNG format) is required by the `-f` or `--adv_dir` argument.
- The output are saved as images (in PNG format) instead of batches of torch tensor or numpy arrays.
- some default arguments are different (e.g. epsilon, surrogate model, etc.)

The setup of defended model follows the CTM family (i.e. [variance tuning](https://github.com/JHL-HUST/VT)). We also use their evaulation scripts (not included in this repo) to perform testing.

Remark: All adversarial outputs are save as PNG format to ensure the lossless property. This is independent to the name of the files, even though they can be named as `.JPG`.

## Testing the Attack Transferability

```
usage: test.py [-h] [--models MODELS] [--batch_size BATCH_SIZE] [-f FOLDER]
               [-b] [-q] [--max_batch MAX_BATCH] [--load_dir LOAD_DIR]

For Testing Attack Transferability

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS       dict_keys(['resnet50', 'wrn50_2', 'inception_v3',
                        'vgg19', 'pnasnet5', 'densenet', 'mobilenet',
                        'resnext101', 'senet50'])
  --batch_size BATCH_SIZE
                        only useful for tests with clean (benign) images
  -f FOLDER, --folder FOLDER
                        the folder to test on
  -b, --benign          whether to test on benign image only
  -q, --quiet           only report the summary
  --max_batch MAX_BATCH
                        the number of batch to break early (default = -1)
  --load_dir LOAD_DIR   The path to load the output tensors for testing
```

To test the adversarial examples generated from ILA-DA in the previous section on all 9 models, run
```
python test.py -f bs100_eps0.03_resnet50_niters10_ila50_l3_1_alpha-1_augatrc_searchgumbel_eta5e-05 --models *
```
The parameter for the mandatory field `-f` should be the exact folder name to test in `data/attack_batches/`.

To test the attack only on VGG19 and Inception V3, run
```
python test.py -f <foldername> --models vgg19/inception_v3
```
The model names are separated by slash `/`.


## Acknowledgements

The following resources are very helpful to our work:

### Pretrained models weights:
ImageNet
- [Torchvision](https://pytorch.org/vision/stable/models.html)
- [PNASNet](https://github.com/Cadene/pretrained-models.pytorch)
- [SENet](https://github.com/moskomule/senet.pytorch)

CIFAR10 & CIFAR100:
- [Bearpaw](https://github.com/bearpaw/pytorch-classification/tree/master/models/cifar)

### Evaluation on Undefended Models:
- [ILA++](https://github.com/qizhangli/ila-plus-plus)
- [LinBP](https://github.com/qizhangli/linbp-attack)

### Evaluation on Defended Models:
- [Variance Tuning](https://github.com/JHL-HUST/VT)
- [Admix](https://github.com/JHL-HUST/Admix)


## Citation

If you find our work helpful in your research, please cite the following:
```
@inproceedings{yan2023ilada,
    title = {ILA-DA: Improving Transferability of Intermediate Level Attack with Data Augmentation},
    author={Yan, Chiu Wai and Cheung, Tsz-Him and Yeung, Dit-Yan},
    booktitle={ICLR},
    year={2023},
}
```
