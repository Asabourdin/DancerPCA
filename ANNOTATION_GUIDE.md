# How to draw the ground-truth masks in Photopea

The 10 frames to annotate are in data/ground_truth/to_annotate/. They are color frames of 640 x 360 pixels, named frame_0001.png, frame_0045.png, and so on. The number is the frame number in the video window (frame 1 = 0:24).

For each frame, we draw every dancer in white on a black background, by hand. Do not start from the masks of DeepLabV3 or of Robust PCA: the ground truth must be fully hand-made, so the comparison stays fair.

What counts as a dancer: the whole body, with hair, clothes and shoes. Do not include shadows on the floor or reflections in the glass wall on the left. If a dancer is cut by the edge of the frame, draw the visible part only.

## Steps for one frame (example: frame_0045.png)

1. Go to https://www.photopea.com. Choose File > Open and pick data/ground_truth/to_annotate/frame_0045.png.

2. Add a drawing layer: Layer > New > Layer. In the Layers panel (on the right), set the Opacity of this new layer to about 60%, so you can still see the dancers through your drawing.

3. Set the drawing color to white: press D (black and white colors), then X (white in front).

4. Trace each dancer with the Polygonal Lasso tool (press L, or Shift+L until you get the polygon icon). Click around the outline of the dancer, point by point, and close the shape by clicking on the first point. Then choose Edit > Fill, Contents: Foreground Color (white), and click OK. Press Cmd+D to deselect. Repeat for every dancer. Zoom in with Cmd and + for small parts like arms and legs.
   For small fixes, use the Brush tool (B) in white to add, or the Eraser (E) to remove. Keep the brush Hardness at 100%.

5. Set the Opacity of the drawing layer back to 100%.

6. Make the photo black: click the photo layer (the bottom one, "Background") in the Layers panel, choose Edit > Fill, Contents: Black, and click OK. You should now see white dancers on a black background.

7. Export: File > Export as > PNG. Keep the size 640 x 360. Name the file frame_0045_mask.png (same number as the frame, then _mask). Save it directly in data/ground_truth/ (not in to_annotate/).

8. Check: open frame_0045_mask.png. It must show only white shapes on black. Gray pixels on the edges are fine: the code counts a pixel as a dancer when it is lighter than mid-gray.

When the 10 masks are in data/ground_truth/, the F1 evaluation runs by itself the next time the code runs.
