# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    color_select= np.copy(image)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1
    theta = np.pi/180*1
    threshold = 15
    min_line_length = 40
    max_line_gap = 20
    line_image = np.copy(image)*0 #creating a blank to draw lines on
    color_image = np.copy(image)*0 #creating a blank to draw lines on
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on the blank
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Define a triangle region of interest 
    left_bottom = [80, 550]
    right_bottom = [900, 550]
    apex = [500, 300]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    ysize = image.shape[0]
    xsize = image.shape[1]
    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    # Define our color criteria
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    # Mask pixels below the threshold
    color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                        (image[:,:,1] < rgb_threshold[1]) | \
                        (image[:,:,2] < rgb_threshold[2])

    line_image[~region_thresholds] = [0,0,0]
    color_image[~color_thresholds & region_thresholds] = [255,0,0]
    final_mask = cv2.add(line_image,color_image)

    merge_mask = np.zeros_like(color_select)
    merge_mask[final_mask[:,:,0] == 255] = 255
    
    # Define our parameters for Canny and apply
    low_threshold = 0
    high_threshold = 255
    edges = cv2.Canny(merge_mask, low_threshold, high_threshold)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    slope_left = 0
    slope_right = 0
    num_left = 0
    num_right = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            #left side line
            if x1 < apex[0] and x2 < apex[0]:
                temp = (y2 - y1) / (x2 - x1 + 0.000001)
                slope_left = slope_left + temp
                num_left = num_left + 1
            #right side line
            if x1 > apex[0] and x2 > apex[0]:
                temp = (y2 - y1) / (x2 - x1 + 0.000001)
                slope_right = slope_right + temp
                num_right = num_right + 1 
    slope_right = slope_right / num_right
    slope_left = slope_left / num_left

    print(slope_right,slope_left)
    #find start point
    start_x_left = 0
    start_x_right = 0
    start_y_left = ysize
    start_y_right = ysize

    for line in lines:
        for x1,y1,x2,y2 in line:
            print(y1,x1)
            if x1 < apex[0] and x2 < apex[0]:
                if start_y_left > y1:
                    start_y_left = y1
                    start_x_left = x1
            if x1 > apex[0] and x2 > apex[0]:
                if start_y_right > y1:
                    start_y_right = y1
                    start_x_right = x1
    end_y = ysize
    end_x_left = int((end_y - start_y_left) / slope_left + start_x_left)
    end_x_right = int((end_y - start_y_right) / slope_right + start_x_right)

    end_x_left = max(0,end_x_left)
    end_x_left = min(xsize,end_x_left)
    end_x_right = max(0,end_x_right)
    end_x_right = min(xsize,end_x_right)

    #unify the y coordinate
    start_y = 300
    start_x_left = int(end_x_left - (end_y - start_y) / slope_left)
    start_x_right = int(end_x_right - (end_y - start_y) / slope_right)

    color_image2 = np.copy(image)*0 

    cv2.line(color_image2,(start_x_left,start_y),(end_x_left,end_y),(255,0,0),10)
    cv2.line(color_image2,(start_x_right,start_y),(end_x_right,end_y),(255,0,0),10)
    result = cv2.addWeighted(image, 0.8, color_image2, 1, 0)
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
