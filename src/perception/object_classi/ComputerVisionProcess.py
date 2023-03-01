import socket
import struct
import time
import numpy as np
import glob



from threading import Thread

import cv2

from src.templates.workerprocess import WorkerProcess

# MAS IMPORTOK

from threading import Thread
import time
from pathlib import Path
import socket

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from src.perception.object_classi.models.experimental import attempt_load
from src.templates.workerprocess import WorkerProcess
from src.perception.object_classi.utils.datasets import LoadStreams, LoadImages
from src.perception.object_classi.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from src.perception.object_classi.utils.plots import plot_one_box
from src.perception.object_classi.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from src.perception.object_classi.utils.datasets import letterbox

class ComputerVisionProcess(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs):
        """Process used for sending images over the network to a targeted IP via UDP protocol 
        (no feedback required). The image is compressed before sending it. 

        Used for visualizing your raspicam images on remote PC.
        
        Parameters
        ----------
        inPs : list(Pipe) 
            List of input pipes, only the first pipe is used to transfer the captured frames. 
        outPs : list(Pipe) 
            List of output pipes (not used at the moment)
        """
        self.img_size = 320
        super(ComputerVisionProcess,self).__init__( inPs, outPs)
        
    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        super(ComputerVisionProcess,self).run()

    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the sending thread.
        """

        laneTh = Thread(name='LaneFindingThread',target = self._lane_finding_thread, args= (self.inPs[0], self.outPs[0],))
        laneTh.daemon = True
        self.threads.append(laneTh)


        
        # objectTh = Thread(name='ObjectDetectionThread',target = self._object_detection_thread, args= (self.inPs[0], ))
        # objectTh.daemon = True
        # self.threads.append(objectTh)

        
    # ===================================== SEND THREAD ==================================
    def _lane_finding_thread(self, inP, outP):
        ###########################################################
        ###########################################################
        ##################                       ##################
        ##################  CAMERA CALIBRATION   ##################
        ##################                       ##################
        ###########################################################
        ###########################################################

        # wE MIGHT WANT TO USE CHECKBOARD IMAGES FOR THIS


        # Advanced Lane Finding Project
        # The goals / steps of this project are the following:

        # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        # Apply a distortion correction to raw images.
        # Use color transforms, gradients, etc., to create a thresholded binary image.
        # Apply a perspective transform to rectify binary image ("birds-eye view").
        # Detect lane pixels and fit to find the lane boundary.
        # Determine the curvature of the lane and vehicle position with respect to center.
        # Warp the detected lane boundaries back onto the original image.
        # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
        # First, I'll compute the camera calibration using chessboard images

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')
        images_with_chessboard_corners = []

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                
                images_with_chessboard_corners.append(img)


        # Horizontal layout
        # https://stackoverflow.com/questions/19471814/display-multiple-images-in-one-ipython-notebook-cell
        # plt.figure(figsize=(50,30))
        # columns = 4
        # for i, image in enumerate(images_with_chessboard_corners):
        #     plt.subplot(len(images_with_chessboard_corners) / columns + 1, columns, i + 1)
        #     plt.imshow(image)

        # Then I will compute camera calibration matrix and distortion coefficients
        def cal_undistort(img, objpoints, imgpoints):
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            
            return undist

        # def plot_two_images(img1, img2, modified_image_text="Undistorted Image"):
        #     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        #     f.tight_layout()
        #     ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        #     ax1.set_title('Original Image', fontsize=50)
        #     ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        #     ax2.set_title(modified_image_text, fontsize=50)
        #     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

            
        # Let's display a few of the chessboard images, but undistorted

        ##### PLOTTING #########
        # Display a few images
        # img = cv2.imread('camera_cal/calibration19.jpg')
        # plot_two_images(img, cal_undistort(img, objpoints, imgpoints))

        # img = cv2.imread('camera_cal/calibration1.jpg')
        # plot_two_images(img, cal_undistort(img, objpoints, imgpoints))

        # img = cv2.imread('camera_cal/calibration10.jpg')
        # plot_two_images(img, cal_undistort(img, objpoints, imgpoints))


        ##### PLOTTING #########
        # Looks good! Now let's apply the same undistortion to some of the test images
        # img = cv2.imread('camera_cal/test1.jpg')
        # plot_two_images(img, cal_undistort(img, objpoints, imgpoints))

        # img = cv2.imread('camera_cal/test2.jpg')
        # plot_two_images(img, cal_undistort(img, objpoints, imgpoints))

        # img = cv2.imread('camera_cal/test3.jpg')
        # plot_two_images(img, cal_undistort(img, objpoints, imgpoints))



        # Time for some gradient threshold.
        # First is Absolute Sobel Threshold

        def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
            
            # Apply the following steps to img
            # 1) Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2) Take the derivative in x or y given orient = 'x' or 'y'
            if (orient == 'x'):
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
                abs_sobel = np.absolute(sobelx)
            
            if (orient == 'y'):
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
                abs_sobel = np.absolute(sobely)

            

            # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
            scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

            
            # 5) Create a mask of 1's where the scaled gradient magnitude 
                    # is > thresh_min and < thresh_max
                    
            binary_output = np.zeros_like(scaled_sobel)
            binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
            # 6) Return this mask as your binary_output image
            #binary_output = np.copy(img) # Remove this line
            return binary_output

        # def plot_threshold_gradient(img1, img2, label="Abs Sobel Threshold"):
        #     # Plot the result
        #     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        #     f.tight_layout()
        #     ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        #     ax1.set_title('Original Image', fontsize=50)
        #     ax2.imshow(img2, cmap='gray')
        #     ax2.set_title(label, fontsize=50)
        #     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


        ##### PLOTTING #########
        # image = cv2.imread('camera_cal/test1.jpg')
        # grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=110)
        # plot_threshold_gradient(image, grad_binary)    

        # image = cv2.imread('camera_cal/test2.jpg')
        # grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=110)
        # plot_threshold_gradient(image, grad_binary)
            
        # image = cv2.imread('camera_cal/test3.jpg')
        # grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=110)
        # plot_threshold_gradient(image, grad_binary)



        # Magnitude Threshold

        def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
            
            # Apply the following steps to img
            # 1) Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 2) Take the gradient in x and y separately
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            
            # 3) Calculate the magnitude 
            
            abs_sobelx = np.absolute(sobelx)
            abs_sobely = np.absolute(sobely)
            gradmag = np.sqrt(sobelx**2 + sobely**2)
            
            
            scale_factor = np.max(gradmag)/255 
            gradmag = (gradmag/scale_factor).astype(np.uint8) 
            # Create a binary image of ones where threshold is met, zeros otherwise
            binary_output = np.zeros_like(gradmag)
            binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
            return binary_output

        ##### PLOTTING #########
        # image = cv2.imread('camera_cal/test5.jpg')
        # mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(90, 200))
        # plot_threshold_gradient(image, mag_binary, "Magnitude Gradient")

        # Direction Threshold
        def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Calculate the x and y gradients
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            # Take the absolute value of the gradient direction, 
            # apply a threshold, and create a binary image result
            absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
            binary_output =  np.zeros_like(absgraddir)
            binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
            return binary_output


        ##### PLOTTING #########
        # image = cv2.imread('camera_cal/test6.jpg')
        # dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.1, 0.6))
        # plot_threshold_gradient(image, dir_binary, "Direction Gradient")

        # Color Threshold
        # Initially I didn't use any color gradient and it was hard for the algorithm to detect the lanes in the area where there were tire marks or shadows, so we need to explore the HLS space to pick out the lane better

        def hls_threshold(image, thresh_l=(160,255), thres_s=(180,255)):
            hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            H = hls[:,:,0]
            L = hls[:,:,1]
            S = hls[:,:,2]

            s_binary = np.zeros_like(S)
            s_binary[(S > thres_s[0]) & (S <= thres_s[1])] = 1

            l_binary = np.zeros_like(L)
            l_binary[(L > thresh_l[0]) & (L <= thresh_l[1])] = 1

            return l_binary, s_binary


        ##### PLOTTING #########

        # image = cv2.imread('camera_cal/test5.jpg')

        # l_binary, s_binary = hls_threshold(image)

        # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        # ax1.set_title('Original Image', fontsize=50)
        # ax2.imshow(s_binary, cmap = 'gray')
        # ax2.set_title('S-channel', fontsize=50)
        # ax3.imshow(l_binary, cmap = 'gray')
        # ax3.set_title('L-Channel', fontsize=50)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        # plt.show()

        def luv_threshold(image, thresh_l=(215,255)):
            print(image.shape)
            luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            L = luv[:,:,0]
            U = luv[:,:,1]
            V = luv[:,:,2]

            l_binary = np.zeros_like(L)
            l_binary[(L > thresh_l[0]) & (L <= thresh_l[1])] = 1

            return l_binary

        image = cv2.imread('camera_cal/test4.jpg')

        l_binary = luv_threshold(image)


        ##### PLOTTING #########
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        # ax1.set_title('Original Image', fontsize=50)
        # ax2.imshow(l_binary, cmap = 'gray')
        # ax2.set_title('LUV - L Channel', fontsize=50)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        # plt.show()

        # Combined gradient
        # Now that we have 3 different types of gradients, let's combine them'

        image = cv2.imread('camera_cal/test6.jpg')
        #image = mpimg.imread('test_images/test3.jpg')
        # Choose a Sobel kernel size

        def combined_gradient(image):
            ksize = 3 # Choose a larger odd number to smooth gradient measurements

            # Apply each of the thresholding functions
            gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh_min=20, thresh_max=110)
            grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh_min=20, thresh_max=110)
            mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(90, 200))
            dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.2, 0.4))
            l_binary, s_binary = hls_threshold(image)
            luv_binary = luv_threshold(image)

            combined = np.zeros_like(dir_binary)
            #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary != 1)) & ((s_binary == 1) & (luv_binary == 1) & (l_binary != 1))] = 1
            combined[ ((gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)) | ((s_binary == 1) | (luv_binary == 1))) ] = 1
            return combined

        ##### PLOTTING #########
        # plot_threshold_gradient(image, combined_gradient(image), "Combined Gradient")

        # Perspective transform
        # Let us now create a perspective transform. We will use the straight lines images from the test directory in order to get the necessary trapezoid coordinates.

        image = cv2.imread('camera_cal/straight_lines1.jpg')
        #image = mpimg.imread('test_images/straight_lines1.jpg')

        # 258, 50 -> lower left
        # 603, 273 -> upper left
        # 681, 273 -> upper right
        # 1041, 50 -> lower right

        ##### PLOTTING #########
        # I used this section to select the points
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.plot(258, 670, ".")
        # plt.plot(590, 450, ".")
        # plt.plot(690, 450, ".")
        # plt.plot(1041, 670, ".")

        src = np.float32([[258, 670], 
                            [590, 450], 
                            [690, 450], 
                            [1041, 670]])
            
        dst = np.float32([[258, 720], 
                        [258, 0], 
                        [1041, 0], 
                        [1041, 720]])

        # Define perspective transform funtion
        def perspective_transform(img):
            img_size = (img.shape[1], img.shape[0])
            
            M = cv2.getPerspectiveTransform(src, dst)
            
            Minv = cv2.getPerspectiveTransform(dst, src)
            
            warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
            
            return warped

        warped_im = perspective_transform(image)


        ##### PLOTTING #########
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        # ax1.set_title = "Source Image"
        # ax1.imshow(image)
        # poly = plt.Polygon(src, closed=True, fill=False, color='#FF0000')
        # ax1.add_patch(poly)

        # ax2.set_title = "Warped Image"
        # ax2.imshow(warped_im)
        # poly = plt.Polygon(dst, closed=True, fill=False, color='#FF0000')
        # ax2.add_patch(poly)
        # <matplotlib.patches.Polygon at 0x1198d4240>

        # Let's see how this looks like on a a curved lane
        image = mpimg.imread('camera_cal/test5.jpg')
        warped_im = perspective_transform(image)


        ##### PLOTTING #########
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        # ax1.set_title = "Source Image"
        # ax1.imshow(image)
        # poly = plt.Polygon(src, closed=True, fill=False, color='#FF0000')
        # ax1.add_patch(poly)

        # ax2.set_title = "Warped Image"
        # ax2.imshow(warped_im)
        # poly = plt.Polygon(dst, closed=True, fill=False, color='#FF0000')
        # ax2.add_patch(poly)
        # <matplotlib.patches.Polygon at 0x11e9ff4a8>

        # Now let's apply this transform on an image with threshold
        #image = cv2.imread('test_images/test6.jpg')
        #image = mpimg.imread('test_images/test3.jpg')
        # Choose a Sobel kernel size
        ksize = 3 # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh_min=20, thresh_max=110)
        grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh_min=20, thresh_max=110)
        mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
        dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        binary_warped = perspective_transform(combined)


        ##### PLOTTING #########
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        # ax1.set_title = "Source Image"
        # ax1.imshow(image)
        # poly = plt.Polygon(src, closed=True, fill=False, color='#FF0000')
        # ax1.add_patch(poly)

        # ax2.set_title = "Warped Image"
        # ax2.imshow(binary_warped, cmap='gray')
        # poly = plt.Polygon(dst, closed=True, fill=False, color='#FF0000')
        # ax2.add_patch(poly)
        # <matplotlib.patches.Polygon at 0x111fb9780>

        # Now let us find the lanes!
        # We will be using a histogram to determine the intensity of the white pixels

        # histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # plt.plot(histogram)
        # [<matplotlib.lines.Line2D at 0x11d3c6550>]

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 60
        # Set minimum number of pixels found to recenter window
        minpix = 40
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        ##### PLOTTING #########
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
        # (720, 0)

        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        ##### PLOTTING #########
        # plt.imshow(result)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
        # (720, 0)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        Minv = cv2.getPerspectiveTransform(dst, src)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        # plt.imshow(result)
        # <matplotlib.image.AxesImage at 0x11e1f4b00>

        # Now let's create a pipeline and feed it multiple images
        # Import everything needed to edit/save/watch video clips
        # from moviepy.editor import VideoFileClip
        # from IPython.display import HTML
        def radius_and_offset(left_fit, right_fit, warped_combination):
            # Define conversions in x and y from pixels space to meters
            ym_per_pix = 30/720 # meters per pixel in y dimension
            xm_per_pix = 3.7/700 # meters per pixel in x dimension
            
            # Fit new polynomials to left (x,y) and right (x,y) pixel points in world space
            # Does it make more sense to test curvature on previous fits or the raw points?
            #left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
            #right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
            y_points = np.linspace(0, warped_combination.shape[0]-1, warped_combination.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            left_fit_cr = np.polyfit(y_points*ym_per_pix, left_fitx*xm_per_pix, 2)
            right_fit_cr = np.polyfit(y_points*ym_per_pix, right_fitx*xm_per_pix, 2)

            # Evaluation point
            #y_eval = np.max(lefty) # bottom of the image
            y_eval = np.max(y_points) # bottom of the image

            # Calculate the left and right curvatures 
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
            
            # Average curvature of the lane
            average_curverad = (left_curverad+right_curverad)/2
            
            '''
            Offset in lane
            '''
            bottom_y = image.shape[0] #image shape = [720,1280]
            left_fit_bottom = left_fit[0]*bottom_y**2 + left_fit[1]*bottom_y + left_fit[2]
            right_fit_bottom = right_fit[0]*bottom_y**2 + right_fit[1]*bottom_y + right_fit[2]

            lane_center = (left_fit_bottom + right_fit_bottom)/2.
            offset_pix = image.shape[1]/2 - lane_center # in pixels, image shape = [720,1280]
            offset_m = offset_pix*xm_per_pix
            return left_curverad, right_curverad, average_curverad, offset_m
        # The final pipeline
        def process_image(img):
            undist = cal_undistort(img, objpoints, imgpoints)
            combined_gradient_binary = combined_gradient(undist)
            
            binary_warped = perspective_transform(combined_gradient_binary)
            
            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(binary_warped.shape[0]//nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set the width of the windows +/- margin
            margin = 60
            # Set minimum number of pixels found to recenter window
            minpix = 40
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                (0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                (0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
            
            # Assume you now have a new warped binary image 
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 60
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
            left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
            left_fit[1]*nonzeroy + left_fit[2] + margin))) 

            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
            right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
            right_fit[1]*nonzeroy + right_fit[2] + margin)))  

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                        ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                        ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            
            # Create an image to draw the lines on
            warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
            # Combine the result with the original image
            result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
            
            # Get the radius of curvature and offset
            left_rad, right_rad, average_rad, offset_m = radius_and_offset(left_fit, right_fit, combined_gradient_binary)
            average_rad_string = "Radius of Curvature: %.2f m" % average_rad
            offset_string = "Center Offset: %.2f m" % offset_m

            print("MOST FIGYELJ : Radius : " + average_rad_string)
            print("S offset az most : " + offset_string)
            
            cv2.putText(result,average_rad_string , (110, 110), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), thickness=2)
            cv2.putText(result, offset_string, (110, 170), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), thickness=2)
            
            return result, average_rad_string, offset_string
            
            
        # Make a list of test images

        while (True):
            success, frame = inP.read()
            image = frame.astype(np.float32)
            cv2.imshow("output", image)

            processed, average_rad_string, offset_string = process_image(frame)

            outP.send([average_rad_string, offset_string])

        # Step through the list and search for chessboard corners
        # for fname in images:
        #     img = cv2.imread(fname)
        #     images_with_lane_marking.append(process_image(img))
            

            
            
            
        # plt.figure(figsize=(50,30))
        # columns = 4
        # for i, image in enumerate(images_with_lane_marking):
        #     plt.subplot(len(images_with_lane_marking) / columns + 1, columns, i + 1)
        #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            

        # Now let's record the video!
        #white_output = 'test_videos_output/project_video_result.mp4'

        #clip1 = VideoFileClip("project_video.mp4").subclip(40,43)
        #clip1 = VideoFileClip("project_video.mp4")

        #white_clip = clip1.fl_image(process_image) # NOTE: this function expects color images!!

        # %time white_clip.write_videofile(white_output, audio=False)


        # [MoviePy] >>>> Building video test_videos_output/project_video_result.mp4
        # [MoviePy] Writing video test_videos_output/project_video_result.mp4
        #   4%|▍         | 51/1261 [00:55<21:53,  1.09s/it]


        # Feeling adventurous, so let's try the challenge video based on the original pipeline
        # white_output = 'test_videos_output/project_video_challenge.mp4'

        # #clip1 = VideoFileClip("project_video.mp4").subclip(20,25)
        # clip1 = VideoFileClip("challenge_video.mp4")

        # white_clip = clip1.fl_image(process_image) # NOTE: this function expects color images!!

        # %time white_clip.write_videofile(white_output, audio=False)
        # [MoviePy] >>>> Building video test_videos_output/project_video_challenge.mp4
        # [MoviePy] Writing video test_videos_output/project_video_challenge.mp4
        # 100%|██████████| 485/485 [03:06<00:00,  3.08it/s]
        # [MoviePy] Done.
        # [MoviePy] >>>> Video ready: test_videos_output/project_video_challenge.mp4 

        # CPU times: user 2min 30s, sys: 45.5 s, total: 3min 16s
        # Wall time: 3min 7s
        # Nope, didn't work :))
 
    def _object_detection_thread(self, inP):
        """Sending the frames received thought the input pipe to remote client by using the created socket connection. 
        
        Parameters
        ----------
        inP : Pipe
            Input pipe to read the frames from CameraProcess or CameraSpooferProcess. 
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        

        while True:
            stamp, image = inP.recv()

            print(type(image))
            print(image.shape)
            print(image.dtype)

            image = image.astype(np.float32)

            print(type(image))
            print(image.shape)
            print(image.dtype)

            img0 = image

            img = letterbox(img0, new_shape = (self.img_size, self.img_size))[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # INNEN IROM BE A DOLGOKAT

            weights = "yolov7-tiny.pt"
            img_size = 320
            conf = 0.3
            device = 'cpu'

            set_logging()
            device = select_device(device)

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = 320 #check_img_size(imgsz, s=stride)  # check img_size

            print("Model loaded")

            classify = False
            if classify:
                modelc = load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names

            print(names)

            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            

            # Csak az image - val kell dolgozzak

            img = torch.from_numpy(img).to(device)
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            print("vigyazz")
            # Inference
            t1 = time_synchronized()

            print("nefelj")
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment='cpu')[0]
            t2 = time_synchronized()

            print("vigyazz")

            # Apply NMS
            pred = non_max_suppression(pred)#, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            print("vigyazz")

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, img0)

            print("eddig")

            for i, det in enumerate(pred):  # detections per image
                s, im0, frame = '', img0

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    save_conf = True

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                        #ezt kell elkuldjem
                        string_to_send = ""
                        string_to_send += ('%g ' * len(line)).rstrip() % line + '\n'
                        print(string_to_send)

                        # Stream results
                        view_img = True
                        if view_img:
                            cv2.imshow(str(p), im0)
                            cv2.waitKey(1)


            
