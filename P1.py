#%% [markdown]
# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---
#%% [markdown]
# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>
#%% [markdown]
# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  
#%% [markdown]
# ## Import Packages

#%%
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## Read in an Image

#%%
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

#%% [markdown]
# ## Ideas for Lane Detection Pipeline
#%% [markdown]
# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**
#%% [markdown]
# ## Helper Functions
#%% [markdown]
# Below are some helper functions to help get you started. They should look familiar from the lesson!

#%%
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    width = img.shape[1]
    height = img.shape[0]
    
    sum_left_slope = 0
    sum_right_slope = 0
    sum_left_x = 0
    sum_left_y = 0
    sum_right_x = 0
    sum_right_y = 0

    avg_right_slope = 0
    avg_right_x = 0
    avg_right_y = 0
    avg_left_slope = 0
    avg_left_x = 0
    avg_left_y = 0

    min_y = height
    max_y = 0

    left_n = 0
    right_n = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            max_y = max([y1, y2, max_y])
            min_y = min([y1, y2, min_y])
            if slope > 0:
                #print("Right m: ", slope)
                sum_right_slope += slope
                sum_right_x += ((x2+x1)/2)
                sum_right_y += ((y2+y1)/2)
                right_n += 1
            elif slope < 0:
                #print("Left m: ", slope)
                sum_left_slope += slope
                sum_left_x += ((x2+x1)/2)
                sum_left_y += ((y2+y1)/2)
                left_n += 1
    
    if right_n > 0:
        avg_right_slope = (sum_right_slope/right_n)
        avg_right_x = (sum_right_x/right_n)
        avg_right_y = (sum_right_y/right_n)

    if left_n > 0:
        avg_left_slope = (sum_left_slope/left_n)
        avg_left_x = (sum_left_x/left_n)
        avg_left_y = (sum_left_y/left_n)

    # print(avg_right_slope, avg_right_x, avg_right_y)
    # print(avg_left_slope, avg_left_x, avg_left_y)

    # Take out the line which is very away from average
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope > 0 and abs(avg_right_slope - slope) > 0.1:
                #print("slope: ", slope)
                sum_right_slope -= slope
                sum_right_x -= ((x2+x1)/2)
                sum_right_y -= ((y2+y1)/2)
                right_n -= 1
            elif slope < 0 and abs(avg_left_slope - slope) > 0.1:
                #print("slope: ", slope)
                sum_left_slope -= slope
                sum_left_x -= ((x2+x1)/2)
                sum_left_y -= ((y2+y1)/2)
                left_n -= 1

    if right_n > 0:
        avg_right_slope = (sum_right_slope/right_n)
        avg_right_x = (sum_right_x/right_n)
        avg_right_y = (sum_right_y/right_n)

    if left_n > 0:
        avg_left_slope = (sum_left_slope/left_n)
        avg_left_x = (sum_left_x/left_n)
        avg_left_y = (sum_left_y/left_n)

    # print(avg_right_slope, avg_right_x, avg_right_y)
    # print(avg_left_slope, avg_left_x, avg_left_y)

    # y = mx + b
    right_b = avg_right_y - avg_right_slope*avg_right_x
    left_b = avg_left_y - avg_left_slope*avg_left_x

    # print("right b: ", right_b)
    # print("left b: ", left_b)

    # Draw the right side lane
    if avg_right_slope != 0:
        right_x1 = int(round((min_y - right_b)/avg_right_slope)) 
        right_x2 = int(round((max_y - right_b)/avg_right_slope))
        if right_x1 > 0.45*width:
            cv2.line(img, (right_x1, min_y), (right_x2, max_y), color, thickness)

    # Draw the left side lane
    if avg_left_slope != 0:
        left_x1 = int(round((min_y - left_b)/avg_left_slope)) 
        left_x2 = int(round((max_y - left_b)/avg_left_slope))
        if left_x1 < 0.55*width:
            cv2.line(img, (left_x1, min_y), (left_x2, max_y), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, (255, 0, 0), 10)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

#%% [markdown]
# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

#%%
import os
import glob
# testImgFiles = os.listdir("test_images/")
testImgFiles = glob.glob("test_images/*")
# print(testImgFiles)

#%% [markdown]
# ## Build a Lane Finding Pipeline
#
# 
#%% [markdown]
# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

#%%
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

#%%
def findLane(img):
    # Read in and grayscale the image
    gray = grayscale(img)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blurGray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blurGray, low_threshold, high_threshold)

    # Define Region of Interest
    imshape = img.shape
    width = imshape[1]
    height = imshape[0]
    delta = 40
    vertices = np.array([[(0,height),(width*0.5-delta, 330), (width*0.5+delta, 330), (width,height)]], dtype=np.int32)
    maskedEdges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 40     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 130 #minimum number of pixels making up a line
    max_line_gap = 150    # maximum gap in pixels between connectable line segments
    lineImg = hough_lines(maskedEdges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    lines_edges = weighted_img(lineImg, img)
    #plt.imshow(lines_edges)
    return lines_edges

testImages = [mpimg.imread(imgFile) for imgFile in testImgFiles]
lineEdageImages = []
n = 1
for i in testImages:
    lines_edges = findLane(i)
    lineEdageImages.append(lines_edges)
    fileName = f"test_images_output/{n:02d}.png"
    n+=1
    mpimg.imsave(fileName, lines_edges)

show_images(lineEdageImages, 2)

#%% [markdown]
# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

#%%
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


#%%
def process_image(img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # Read in and grayscale the image
    gray = grayscale(img)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blurGray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blurGray, low_threshold, high_threshold)

    # Define Region of Interest
    imshape = img.shape
    width = imshape[1]
    height = imshape[0]
    delta = 40
    vertices = np.array([[(0,height),(width*0.5-delta, 330), (width*0.5+delta, 330), (width,height)]], dtype=np.int32)
    maskedEdges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 40     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 130 #minimum number of pixels making up a line
    max_line_gap = 150    # maximum gap in pixels between connectable line segments
    lineImg = hough_lines(maskedEdges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    result = weighted_img(lineImg, img)

    return result

#%% [markdown]
# Let's try the one with the solid white lane on the right first ...

#%%
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')

#%% [markdown]
# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

#%%
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

#%% [markdown]
# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**
#%% [markdown]
# Now for the one with the solid yellow lane on the left. This one's more tricky!

#%%
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


#%%
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

#%% [markdown]
# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 
#%% [markdown]
# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

#%%
challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
#clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


#%%
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


