# Train Guide

## 1. Prepare dataset
Here is an example of how to prepare [VKITTI2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) dataset.
Download the dataset and save it to `/path/to/VKITTI2`, and then run 
```shell
cd data/VKITTI2
python generate_json_VKITTI2.py
```

## 2. Download base models
Download the models from huggingface and save them into `pretrained` folder:
- [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
```shell
pip install huggingface_hub

mkdir pretrained
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir pretrained/sd-vae-ft-mse --local-dir-use-symlinks False
huggingface-cli download lambdalabs/sd-image-variations-diffusers --local-dir pretrained/image_encoder --local-dir-use-symlinks False
huggingface-cli download stabilityai/stable-diffusion-2-1 --local-dir pretrained/stable-diffusion-2-1 --local-dir-use-symlinks False
```


## 3. Modify the `configs/train.yaml` file
- `meta_paths`: you can also add more datasets
- `base_model_path`: stable diffusion 2.1 for finetune
- `vae_model_path`: freeze
- `image_encoder_path`: freeze

## 4. Train
```shell
python train.py  #  for a single GPU
accelerate launch --mixed_precision="fp16" --num_processes=2 train.py  # for multiple GPUs
```