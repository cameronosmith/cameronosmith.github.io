#!/bin/bash
for filename in */*.mp4; do
    echo $filename;
    cp ../codecfix_img/"$filename" "$filename"
    #ffmpeg -i "$filename" -pix_fmt yuv420p -crf 18 ../codecfix_img/"$filename"
done
