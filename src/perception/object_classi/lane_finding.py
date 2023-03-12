import cv2
import numpy as np

def denoise_frame(frame):
    kernel = np.ones((3,3), np.float32) / 9
    denoised = cv2.filter2D(frame, -1, kernel)
    return denoised

def apply_Canny(frame):
    #frame = denoise_frame(frame)
    grayscaled =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(grayscaled, 50, 150)
    return canny_edges

def region_of_interest(frame):
    #edges = apply_Canny(frame)
    height, width = frame.shape
    #print(f"HEIGHT {height}\nWIDHT {width}")
    mask = np.zeros_like(frame)

    # [[65, 283],  -> COORDS FOR THE CAR BFMC
    # [135, 169], 
    # [230, 169], 
    # [320, 278]]

    # polygon = np.array([[
    #     (25, 283),              # Bottom-left point
    #     (105,  169),   # Top-left point
    #     (230, 169),    # Top-right point
    #     (320, 278),              # Bottom-right point
    #     ]], np.int32)
    # cv2.fillPoly(mask, polygon, 255)

    polygon = np.array([[
        (0, int(height*0.9)),              # Bottom-left point
        (int(width*0.30),  int(height*0.55)),   # Top-left point
        (int(width*0.70), int(height*0.55)),    # Top-right point
        (int(width), int(height*0.9)),              # Bottom-right point
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    #print(frame.size, mask.size)

    cropped_edges = cv2.bitwise_and(frame, mask)

    return cropped_edges


def warp_perspective(frame):
    height, width = frame.shape
    # Offset for frame ratio saving
    offset = 0    
    
    # # Perspective points to be warped
    # source_points = np.float32([[105, 169],     # Top-left point
    #                   [230, 169],               # Top-right point
    #                   [25, 283],                # Bottom-left point
    #                   [320, 278]])              # Bottom-right point
    
    # # Window to be shown
    # destination_points = np.float32([[25, 0],   # Top-left point
    #                   [320, 0],                 # Top-right point
    #                   [25, 320],                # Bottom-left point
    #                   [320, 320]])              # Bottom-right point

    source_points = np.float32([[int(width*0.35), int(height*0.45)], # Top-left point
                      [int(width*0.65), int(height*0.45)],           # Top-right point
                      [0, int(height*0.9)],                     # Bottom-left point
                      [int(width), int(height*0.9)]])                    # Bottom-right point
    
    destination_points = np.float32([
        [offset, 0],                             # Top-left point
        [width-2*offset, 0],                     # Top-right point
        [offset, height],                        # Bottom-left point
        [width-2*offset, height]])               # Bottom-right point
    
    # Matrix to warp the image for skyview window
    matrix = cv2.getPerspectiveTransform(source_points, destination_points) 
    
    # Final warping perspective 
    skyview = cv2.warpPerspective(frame, matrix, (width, height))    

    return skyview    # Return skyview frame


def histogram(frame):
    histogram = np.sum(frame, axis=0)   

    midpoint = int(histogram.shape[0]/2)    
    
    # Compute the left max pixels
    left_x_base = np.argmax(histogram[:midpoint])   
    
    # Compute the right max pixels
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint  
    
    return left_x_base, right_x_base    # Return left_x and right_x bases

def detect_lines(frame):    
    # Find lines on the frame using Hough Lines Polar
    line_segments = cv2.HoughLinesP(frame, 1, np.pi/180 , 20, 
                                    np.array([]), minLineLength=40, maxLineGap=150)
    return line_segments    # Return line segment on road


def optimize_lines(frame, lines):
    height, width, _ = frame.shape  # Take frame size
    
    if lines is not None:   # If there no lines we take line in memory
        # Initializing variables for line distinguishing
        lane_lines = [] # For both lines
        left_fit = []   # For left line
        right_fit = []  # For right line
        
        for line in lines:  # Access each line in lines scope
            x1, y1, x2, y2 = line.reshape(4)    # Unpack actual line by coordinates

            parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Take parameters from points gained
            slope = parameters[0]       # First parameter in the list parameters is slope
            intercept = parameters[1]   # Second is intercept

            horiz_slope = 0.15
            
            if slope < -horiz_slope:   # Here we check the slope of the lines 
                left_fit.append((slope, intercept))
            elif slope > horiz_slope:   
                right_fit.append((slope, intercept))

        if len(left_fit) > 0:       # Here we ckeck whether fit for the left line is valid
            left_fit_average = np.average(left_fit, axis=0)     # Averaging fits for the left line
            lane_lines.append(map_coordinates(frame, left_fit_average)) # Add result of mapped points to the list lane_lines
        else:
            lane_lines.append(None)
            
        if len(right_fit) > 0:       # Here we ckeck whether fit for the right line is valid
            right_fit_average = np.average(right_fit, axis=0)   # Averaging fits for the right line
            lane_lines.append(map_coordinates(frame, right_fit_average))    # Add result of mapped points to the list lane_lines
        else:
            lane_lines.append(None)
    else:
        return [None, None]
        
    return lane_lines       # Return actual detected and optimized line 

def map_coordinates(frame, parameters):
    height, width, _ = frame.shape  # Take frame size
    slope, intercept = parameters   # Unpack slope and intercept from the given parameters
    
    if abs(slope) < 0.1:      # Check whether the slope is 0
        slope = 0.1     # handle it for reducing Divisiob by Zero error
    
    y1 = height             # Point bottom of the frame
    y2 = int(height*0.72)  # Make point from middle of the frame down  
    x1 = int((y1 - intercept) / slope)  # Calculate x1 by the formula (y-intercept)/slope
    x2 = int((y2 - intercept) / slope)  # Calculate x2 by the formula (y-intercept)/slope
    
    return [[x1, y1, x2, y2]]   # Return point as array

def display_lines(frame, lines):
    mask = np.zeros_like(frame)   
    
    if lines is not None:                   # Check if there is a existing line
        for line in lines:                  # Iterate through lines list
            for x1, y1, x2, y2 in line:     # Unpack line by coordinates
                cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 2)    # Draw the line on the created mask
    
    frame = cv2.addWeighted(frame, 0.8, mask, 1, 1)    
    
    return frame    # Return frame with displayed lines

def get_floating_center(frame, lane_lines):    
    height, width, _ = frame.shape # Take frame size
    
    if len(lane_lines) == 2:    # Here we check if there is 2 lines detected
        left_x1, ly1, left_x2, ly2 = lane_lines[0][0]   # Unpacking left line
        right_x1, ry1, right_x2, ry2 = lane_lines[1][0] # Unpacking right line
        
        low_mid = (right_x1 + left_x1) / 2  # Calculate the relative position of the lower middle point
        up_mid = (right_x2 + left_x2) / 2

        ly = (ry1 + ly1) / 2
        uy = (ry2 + ly2) / 2

        cv2.line(frame, (int(low_mid), int(ly)), (int(up_mid), int(uy)), (255, 0, 0), 2)

    else:       # Handling undetected lines
        up_mid = int(width*1.9)
        low_mid = int(width*1.9)

    
    return up_mid, low_mid       # Return shifting points

def add_text(frame, image_center, left_x_base, right_x_base):
    print("left base:", left_x_base)
    print("right base:", right_x_base)
    print("image_center:", image_center)
    

    lane_center = left_x_base + (right_x_base - left_x_base) / 2 # Find lane center between two lines
    
    deviation = image_center - lane_center    # Find the deviation
 
    if deviation > 160:         # Prediction turn according to the deviation
        text = "Smooth Left"
    elif deviation < 40 or deviation > 150 and deviation <= 160:
        text = "Smooth Right"
    elif deviation >= 40 and deviation <= 150:
        text = "Straight"
    
    cv2.putText(frame, "DIRECTION: " + text, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA) # Draw direction
    
    return deviation, frame    # Retrun frame with the direction on it

def process_frame(frame):
    edges = apply_Canny(frame)

    denoised_frame = denoise_frame(frame)   # Denoise frame from artifacts

    canny_edges = apply_Canny(denoised_frame)  # Find edges on the frame

    roi_frame = region_of_interest(canny_edges)   # Draw region of interest

    warped_frame = warp_perspective(canny_edges)    # Warp the original frame, make it skyview

    # cv2.imshow('out', warped_frame)
    # cv2.waitKey()

    left_x_base, right_x_base = histogram(warped_frame)         # Take x bases for two lines
    lines = detect_lines(roi_frame)                 # Detect lane lines on the frame
    lane_lines = optimize_lines(frame, lines)       # Optimize detected line
    # print("lane_lines:")
    # print(lane_lines)

    if lane_lines == [None, None]:
        return 0, frame
    elif lane_lines[0] == None: # nincs bal
        return -700, frame
    elif lane_lines[1] == None: # nincs jobb
        return 700, frame

    
    lane_lines_image = display_lines(frame, lane_lines) # Display solid and optimized lines
    
    up_center, low_center = get_floating_center(frame, lane_lines) # Calculate the center between two lines

    # if lane_lines == [None, None]:
    #     return 0, frame
    # elif lane_lines[0] == None: # nincs bal
    #     deviation, final_frame = add_text(lane_lines_image, low_center, left_x_base, right_x_base) # Predict and draw turn
    #     return -700, final_frame
    # elif lane_lines[1] == None: # nincs jobb
    #     deviation, final_frame = add_text(lane_lines_image, low_center, left_x_base, right_x_base) # Predict and draw turn
    #     return 700, final_frame

    deviation, final_frame = add_text(lane_lines_image, low_center, left_x_base, right_x_base) # Predict and draw turn

    return deviation, final_frame  # Return final frame

# img = cv2.imread('camera_cal/kep2.jpg')
# prd = process_frame(img)
# cv2.imshow('out', prd)
# cv2.waitKey()