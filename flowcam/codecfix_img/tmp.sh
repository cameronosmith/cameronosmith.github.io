#!/bin/bash
for filename in */*.mp4; do
    echo $filename;
    #ffmpeg -i $filename -pix_fmt yuv420p -crf 18 $filename
done
