# GenStereo: Towards Open-World Generation of Stereo Images and Unsupervised Matching


<div class="is-size-5 publication-authors">
  <span class="author-block">
    <a href="https://qjizhi.github.io/" target="_blank">Feng Qiao</a>,
  </span>
  <span class="author-block">
    <a href="https://steven-xiong.github.io/" target="_blank">Zhexiao Xiong</a>,
  </span>
  <span class="author-block">
    <a href="https://ericx003.github.io/" target="_blank">Eric Xing</a>,
  </span>
  <span class="author-block">
    <a href="https://jacobsn.github.io/" target="_blank">Nathan Jacobs</a>
  </span>
</div>
<div class="is-size-5 publication-authors">
  <span class="author-block">Washington University in St. Louis</span>
</div>


[![Project Site](https://img.shields.io/badge/Project-Web-green)](https://qjizhi.github.io/genstereo) &nbsp;
[![Spaces](https://img.shields.io/badge/Spaces-Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/FQiao/GenStereo) &nbsp;
[![Github](https://img.shields.io/badge/Github-Repo-orange?logo=github)](https://github.com/Qjizhi/GenStereo) &nbsp;
[![Models](https://img.shields.io/badge/Models-checkpoints-blue?logo=huggingface)](https://huggingface.co/FQiao/GenStereo-sd2.1/tree/main) &nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2503.12720-red?logo=arxiv)](https://arxiv.org/abs/2503.12720)

[Introduction](#introduction)
| [Demo](#demo)
| [How to use](#how-to-use)
| [Citation](#citation)
| [Acknowledgements](#acknowledgements)

![concept image](./images/teaser_coco.jpg)

## Updates

- **06/02/2025:** Release GenStereo v2.1, which demonstrates better performance and higher resolution! Check out the [demo](https://huggingface.co/spaces/FQiao/GenStereo).
- **03/17/2025:** Codes and demos are released!

## Introduction

This repository is an official implementation for the paper "[GenStereo:  Towards Open-World Generation of Stereo Images and Unsupervised Matching](https://qjizhi.github.io/genstereo/)". Given an arbitrary reference image, GenStereo generates the corresponding right-view image by enforcing constraints at three levels: input (disparity-aware coordinate and warped-image embeddings), feature (cross-view attention), and output (pixel-level loss with adaptive fusion). These constraints yield stereo images with geometric consistency and visual quality. Our methods demonstrate state-of-the-art performance in both stereo image generation and unsupervised stereo matching.

![Framework](./images/framework.jpg)

## Demo

Try the demo [here](https://huggingface.co/spaces/FQiao/GenStereo).

## How to use

### Environment

We tested our codes on Ubuntu with nVidia A100 GPU. If you're using other machines like Windows, consider using Docker. You can either add packages to your python environment or use Docker to build an python environment. Commands below are all expected to run in the root directory of the repository.

We tested the environment with python `>=3.10` and CUDA `=11.8`. To add mandatory dependencies run the command below.

``` shell
pip install -r requirements.txt
```

To run developmental codes such as the example provided in jupyter notebook and the live demo implemented by gradio, add extra dependencies via the command below.

``` shell
pip install -r requirements_dev.txt
```

### Download pretrained models

GenStereo uses pretrained models which consist of both our finetuned models and publicly available third-party ones. Download all the models to `checkpoints` directory or anywhere of your choice. You can do it manually or by the [download_models.sh](scripts/download_models.sh) script.

#### Download script

``` shell
bash scripts/download_models.sh 
```

#### Manual download

> [!NOTE]
> Models and checkpoints provided below may be distributed under different licenses. Users are required to check licenses carefully on their behalf.

1. Our finetuned models, we provide two versions of GenStereo
    - v1.5: 512px, faster, [model card](https://huggingface.co/FQiao/GenStereo).
    - v2.1: 768px, better performance, high resolution, takes more time, [model card](https://huggingface.co/FQiao/GenStereo-sd2.1).
2. Pretrained models:
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
      - download `config.json` and `diffusion_pytorch_model.safetensors` to `checkpoints/sd-vae-ft-mse`
    - [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
      - download `image_encoder/config.json` and `image_encoder/pytorch_model.bin` to `checkpoints/image_encoder`
3. MDE (Monocular Depth Estimation) models
    - We use [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) as the MDE model and get the disparity maps.
The final `checkpoints` directory must look like this:

```
.
├── depth_anything_v2_vitl.pth
├── genstereo-v1.5
│   ├── config.json
│   ├── denoising_unet.pth
│   ├── fusion_layer.pth
│   ├── pose_guider.pth
│   └── reference_unet.pth
├── genstereo-v2.1
│   ├── config.json
│   ├── denoising_unet.pth
│   ├── fusion_layer.pth
│   ├── pose_guider.pth
│   └── reference_unet.pth
├── image_encoder
│   ├── config.json
│   └── pytorch_model.bin
└── sd-vae-ft-mse
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

### Inference
You can easily run the inference code by running the following command, and the results will be save under `./vis` folder.

```bash
python test.py /path/to/your/image
```


### Gradio live demo

An interactive live demo is also available. Start gradio demo by running the command below, and goto [http://127.0.0.1:7860/](http://127.0.0.1:7860/)
If you are running it on the server, be sure to forward the port 7860.

Or you can just visit [Spaces](https://huggingface.co/spaces/FQiao/GenStereo) hosted by Hugging Face to try it now.

```shell
python app.py
```

## Citation

``` bibtex
  @article{qiao2025genstereo,
    author    = {Feng Qiao and Zhexiao Xiong and Eric Xing and Nathan Jacobs},
    title     = {GenStereo: Towards Open-World Generation of Stereo Images and Unsupervised Matching},
    journal   = {arXiv preprint},
    volume    = {arXiv:2503.12720},
    year      = {2025},
    url       = {https://doi.org/10.48550/arXiv.2503.12720},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
  }
```

## Acknowledgements

Our codes are based on [GenWarp](https://github.com/sony/genwarp),  [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) and other repositories. We thank the authors of relevant repositories and papers.
