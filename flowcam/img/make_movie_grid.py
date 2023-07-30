base_path="dummy-Gi6LJcRrny2kqR548cR2u4_Jul302023_681"
num_scene=20
n_per_row=min(num_scene,2)
from moviepy.editor import VideoFileClip, clips_array,concatenate_videoclips,ColorClip, vfx,CompositeVideoClip,TextClip
from glob import glob
from moviepy.editor import ImageSequenceClip
from moviepy.video.fx.all import speedx,scroll
import numpy as np

clips=[]
for scene_i in range(num_scene):
    for x in ["input_video_%03d.mp4"%scene_i, "time_interp_%03d.mp4"%scene_i, "wobble_%03d.mp4"%scene_i]:
        clips.append(VideoFileClip(base_path+"/"+x))
    image_paths = list(sorted(glob("%s/%d/*"%(base_path,scene_i))))
    clips.append(ImageSequenceClip(image_paths+image_paths[::-1], fps=8).resize(clips[-1].size))

def stretch_clip(clip, new_duration,max_height):
    current_duration = clip.duration
    speed_factor = current_duration/new_duration 
    stretched_clip =  speedx(clip, speed_factor)

    # Calculate height difference between the stretched clip and the maximum height
    height_diff = max(max_height-stretched_clip.h, 0)
    print(height_diff)
    
    # Pad the stretched clip only at the bottom with a black background
    if height_diff > 0 and 0:
        padding = ColorClip((stretched_clip.size[0],stretched_clip.size[0]), color=(255,255,255), duration=stretched_clip.duration)
        padding = padding.set_duration(new_duration)
        final_clip = clips_array([[stretched_clip],[padding]])
        return final_clip
    else:
        return stretched_clip

max_duration = max(clip.duration for clip in clips)
max_height = max(clip.h for clip in clips)

titles = []
for _ in range(n_per_row): titles+=[ "Input Video", "Interp.", "Interp.+Wobble", "Poses" ] 

# Create ColorClip objects for each title
title_clips = []
for title in titles:
    title_clip = ColorClip(size=(clips[0].w, 15), color=(255,255,255), duration=max_duration)
    title_text = TextClip(title, fontsize=13, color='black', bg_color='white',font="Arial")
    title_text = title_text.set_duration(max_duration)
    title_text = title_text.set_position(("center", "top"))
    title_clips.append(title_text)

clips = [stretch_clip(clip,max_duration,max_height) for clip in clips]
clips = [title_clips]+np.array(clips).reshape(-1,n_per_row*4).tolist()
clip = clips_array(clips,bg_color=(255,255,255))
clip.subclip(0, clip.duration - .5).write_videofile(base_path+".mp4",codec="libx264", audio_codec="aac", fps=30, bitrate="5000k")
