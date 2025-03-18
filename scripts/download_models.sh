#!/bin/bash

ckpt_dir=${1:-./checkpoints}

function download () {
    out_dir=${ckpt_dir}/$1
    uri_base=$2
    shift 2
    mkdir -p "$out_dir"
    for i in "$@"; do
        uri=${uri_base}/$i
        fn=${out_dir}/$i
        echo "Downloading $fn..."
        if [ -f "$fn" ]; then
            local=$(wc -c < "$fn")
            remote=$(curl -sIL "$uri" | awk 'BEGIN{IGNORECASE = 1}/Content-Length/ {sub("\r",""); print $2}' | tail -1)
            if [ "$local" = "$remote" ]; then
                continue
            fi
        fi
        curl -sL "$uri" -o "$fn"
    done
}

download \
    image_encoder \
    https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder \
    config.json \
    pytorch_model.bin

download \
    sd-vae-ft-mse \
    https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main \
    config.json \
    diffusion_pytorch_model.safetensors

download \
    genstereo \
    https://huggingface.co/FQiao/GenStereo/resolve/main \
    config.json \
    denoising_unet.pth \
    fusion_layer.pth \
    pose_guider.pth \
    reference_unet.pth

# Additional file download using wget to the ckpt_dir
depth_anything_file="${ckpt_dir}/depth_anything_v2_vitl.pth"
depth_anything_url="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"

echo "Downloading ${depth_anything_file}..."
wget -q "${depth_anything_url}" -O "${depth_anything_file}"
