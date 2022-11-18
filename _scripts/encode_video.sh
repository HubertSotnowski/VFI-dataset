#!/bin/bash

/usr/bin/ffmpeg \
    -i $1 \
    -c:v hevc_nvenc \
    -pix_fmt yuv420p \
    -vf scale=1920:1080 \
    -rc constqp \
    -qp 10 \
    -preset medium \
    -an \
    -sn \
    $2

