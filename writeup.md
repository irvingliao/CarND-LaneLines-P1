# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

Pipeline:
1. Read in and grayscale the image
2. Define the Gaussian kernel size to 5 and apply Gaussian smoothing
3. Apply Canny low_threshold = 50, high_threshold = 150
4. Define Region of Interest, I set the Quadrilateral area around the y=330 and middle of the image with +- 45 pixel range.
5. Finally define the Hough transform parameters and draw the lane line

draw_lines() modification:
Original thoght is to segment the lines with positive or negative slope value, and right lane most likely will be on right half area of camera, and left lane will be mostly on left half area. Then take the average of slope and (x, y) so we would conclude with a very close lane line.
However, in some case, some of the lines will be very away from the average value, and it would be impact significiant to the final slope or (x, y) values. So I'll try to take those very away lines out from the calculation.


### 2. Identify potential shortcomings with your current pipeline
A potential issue with the pipeline is if there's many cars or shadows or signs on the road, or even soem random objects in the road, it would be possible mess up with the lane detection.

And for lanes that have a big turn angle, which possibly will not function too.


### 3. Suggest possible improvements to your pipeline

There's should be a way to identify the objects that are differnt from the lanes.
Probably we can identify the shapes and continuous context before and after, so if there's a random object we detect which is not exist before, which shown suddenly. I would know it would not the lanes we are looking for.
