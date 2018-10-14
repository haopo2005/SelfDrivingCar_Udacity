#image undistortion
import pickle
import cv2
import numpy as np

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_images/test5.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('output_images/test5_udist.jpg',undist)


# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary ==1)|(sxbinary ==1)] = 1
    return combined_binary

def calculate_curvature(ploty, leftx, rightx):
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    offset = ((leftx[-1] + rightx[-1]) / 2 - 640) * xm_per_pix
    return left_curverad, right_curverad, offset

def print_stats(image, c_l, c_r, diff):
    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(image,
                'Estimated curvature: Left: {0}m, Right: {1}m.'.format(round(c_l, 2),
                                                                       round(c_r, 2)),
                (10, 50), font, 1, (255, 255, 255), 2)

    cv2.putText(image, 'Offset from lane center : {0}m'.format(round(diff, 2)),
                (10, 150), font, 1, (255, 255, 255), 2)

threshold_img = pipeline(undist)*255
cv2.imwrite('output_images/threshold_img.jpg',threshold_img)

src = np.float32([[610, 450], [720, 450], [1200, 720], [280, 720]])
dst = np.float32([[350, 0], [900, 0], [900, 720], [350, 720]])

img_size = (img.shape[1], img.shape[0])

M = cv2.getPerspectiveTransform(src, dst)
binary_warped = cv2.warpPerspective(threshold_img, M, img_size, flags=cv2.INTER_LINEAR)
cv2.imwrite('output_images/warped.jpg',binary_warped)

#沿水平方向求直方图
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:],axis=0)
#构造画布，将堆叠三个通道的binary_warped图像
out_img = np.dstack((binary_warped,binary_warped,binary_warped))

#寻找直方图的两个波峰
midpoint = np.int(histogram.shape[0]/2) #直方图的中点，上图来看中点正好可以分割两个波峰
leftx_base = np.argmax(histogram[:midpoint]) #左侧起点
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

#垂直堆叠的滑窗层数
nwindows = 10
#设定滑窗的高度
window_height = np.int(binary_warped.shape[0]/nwindows)
#设定滑窗的宽度
margin = 100
#获取binary_warped图像中所有非零像素点的坐标
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

#当前波峰位置
leftx_current = leftx_base
rightx_current = rightx_base

#滑窗阈值，即若小于这个阈值，则此滑窗位置不应该是波峰的位置
minpix = 50

left_lane_inds = []
right_lane_inds = []

#挨个处理滑窗
for window in range(nwindows):
    #确定每一层滑窗内道路线的y坐标搜索范围，也是滑窗顶点坐标
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height

    #确定每一层滑窗内左右道路线的波峰x坐标搜索范围，也是滑窗顶点坐标
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    
    #画出左右滑窗
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
    
    #识别当前滑窗内的非零像素坐标
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    
    #append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    
    #若当前滑窗内非零像素大于阈值，则更新波峰坐标，否则沿用之前的，垂直堆叠
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
#concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)


#获取左右滑窗内所有非零像素坐标
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

#二次拟合
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

#visualization
#You're fitting for f(y), rather than f(x),
#because the lane lines in the warped image are near vertical and
#may have the same x value for more than one y value
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# calculate the curvature
left_curvrad, right_curvrad, offset = calculate_curvature(ploty, left_fitx, right_fitx)

out_img[nonzeroy[left_lane_inds],nonzerox[left_lane_inds]] = [255,0,0]
out_img[nonzeroy[right_lane_inds],nonzerox[right_lane_inds]] = [0,0,255]
   

# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))],np.int32)
pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))],np.int32)

cv2.polylines(out_img, pts_left, False, (0, 255, 255), 15)
cv2.polylines(out_img, pts_right, False, (0, 255, 255),15)
cv2.imwrite('output_images/out_img.jpg',out_img)

pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))
# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))


Minv = cv2.getPerspectiveTransform(dst, src)

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
# print stats
print_stats(result, left_curvrad, right_curvrad, offset)

cv2.imwrite('output_images/result.jpg', result)