# NISSL PROJECT 

 Nissl cell segmentation using Residual UNet with Attention

## Installation

The code uses pytorch framework torch >= 1.8 and cuda >= 10.0.

Clone the repo and execute to following commands for training and testing
### Cloning the repo
```bash
git clone https://github.com/mallepallihimasagar/NISSL-RESIDUAL_ATTENTION_UNET.git
cd NISSL-RESIDUAL_ATTENTION_UNET
```
Trained weights for [Residual UNet with Attention](https://drive.google.com/file/d/1-1iFeZkkYfNyGyCBgU5kydtwgoJS408I/view?usp=sharing) and [UNet](https://drive.google.com/file/d/1ov4Vzcgj8LvLDZyIanBv34hHYiwcDYei/view?usp=sharing) 
### Training the model
```bash
python train.py --model_name <model name> --num_epochs <no.of epochs> --model_save_path <filename.pt>

## Example
python train.py --model_name 'res_att_unet' --num_epochs 100 --model_save_path resunet.pt
```
- For using UNet Architecture replace <model name> with **'unet'**.
- <model name> **'res_att_unet'** for Residual UNet with Attention Architecture.

If you want to use pre-trained weights and train the model use the following command:
```bash
python train.py --model_name <model name> --num_epochs <no.of epochs> --model_save_path <filename.pt> --weights <path to pretraind weights (.pt file)>
```

### Testing the model
```bash
python test.py --model_name <model name> --testset_path <testset path> --output <out folder> --weights <trained weights>

## Example
python test.py --model_name res_att_unet --testset_path Nissl_Dataset/test --output resattunet_out --weights resunet.pt
```
