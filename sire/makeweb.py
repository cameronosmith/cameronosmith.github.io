import os
from glob import glob
textsrc=("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlowMap++ Visualizations</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .video-grid {
            display: grid;
            grid-template-columns: auto repeat(5, 1fr); /* Label column + 4 video columns */
            gap: 20px;
            align-items: center;
        }
        .video-cell {
            text-align: center;
        }
        .video-row-label {
            writing-mode: vertical-rl;
            text-align: center;
            font-weight: bold;
            font-size: 18px;
            color: #333;
        }
        video {
            width: 100%;
            height: 200px;
            #height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .video-text {
            margin: 10px 0;
            font-size: 14px;
            color: #555;
        }
    </style>
</head>

<body>
    <div class="container mt-4">
        <h1 class="text-center mb-4">FlowMap++ Visualizations</h1>
        <div class="video-grid">
"""
)
for scene in glob("vid_exps/*"):
    scenename=scene.split("/")[-1]
    textsrc+= f"""
            <!-- Scene {scenename} -->
            <div class="video-row-label">{scenename}</div>
            <div class="video-cell">
                <video autoplay muted loop controls>
                    <source src="./{scene}/data_vis.mp4" >
                </video>
                <p class="video-text">Input Data Vis <br> Scene 1</p>
            </div>
            <div class="video-cell">
                <video autoplay muted loop controls>
                    <source src="./{scene}/flowmap_opt.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p class="video-text">FlowMap Optimization <br> 2 min</p>
            </div>
            <div class="video-cell">
                <video autoplay muted loop controls>
                    <source src="./{scene}/flowmap_render.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p class="video-text">FlowMap Output <br> <a href="tmp"> Open in Interactive Viewer </a></p>
            </div>
            <div class="video-cell">
                <video autoplay muted loop controls>
                    <source src="./{scene}/splat_opt.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p class="video-text">Splatting Optimization <br> 2 minutes</p>
            </div>
            <div class="video-cell">
                <video autoplay muted loop controls>
                    <source src="./{scene}/splat_render.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p class="video-text">Splatting Output <br> <a href="tmp"> Open in Interactive Viewer </a></p>
            </div>
"""
textsrc+= """
</div>
</div>
</body>
</html>
"""
print(textsrc)
