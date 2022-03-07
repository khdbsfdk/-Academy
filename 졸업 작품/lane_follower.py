# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import sys
import time
import serial
import RPi.GPIO as GPIO

def rescale_frame(frame,percent):
        width = int(frame.shape[1]*percent/100)
        height = int(frame.shape[0]*percent/100)
        dim = (width,height)
        return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

def convert_gray(frame):

  blur = cv2.GaussianBlur(frame,(5,5),0)

  gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

  #equal = cv2.equalizeHist(gray)

  return gray

def detect_edges(frame):

    edges = cv2.Canny(frame, 200, 400)
    return edges

def region_of_interest(edges):

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (0,  height/3),
        (width , height/3),
        (width, height),
    ]], np.int32)
 
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

 

def detect_line_segments(cropped_edges):

    rho = 1
    theta = np.pi/180
    min_threshold = 30
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                         np.array([]), minLineLength=8, maxLineGap=4)
    return line_segments

 

def average_slope_intercept(frame, line_segments):

    lane_lines = []
    if line_segments is None:
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1-boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 -(slope * x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))

            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

 

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line

    y1 = height  
    y2 = int(y1 / 2) 
 
    if slope == 0:
        slope = 0.1

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]

 

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):

    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):

    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    steering_angle_radian = steering_angle / 180.0 * math.pi

    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

def get_steering_angle(frame, lane_lines):

    height,width,_ = frame.shape
        
    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)
            
    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)
        
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90
                
    return steering_angle

try:
    
    video = cv2.VideoCapture(-1)

    pin=18
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin,GPIO.OUT)
    p=GPIO.PWM(pin,50)
    p.start(0)
     
    if video.isOpened():
        while True:
            
            ret,frame = video.read()

            if ret:
                frame = rescale_frame(frame,50)
                #frame = cv2.flip(frame,-1)
                hsv = convert_gray(frame)
                edges = detect_edges(hsv)
                roi = region_of_interest(edges)
                line_segments = detect_line_segments(roi)
                lane_lines = average_slope_intercept(frame,line_segments) 
                lane_lines_image = display_lines(frame,lane_lines)
                steering_angle = get_steering_angle(frame, lane_lines)
                heading_image = display_heading_line(lane_lines_image,steering_angle)
                cv2.imshow("heading line",heading_image)

                if steering_angle >= 115:
                    steering_angle = 115
                elif steering_angle <= 30:
                    steering_angle = 30
                elif (steering_angle >= 85 and steering_angle <= 95):
                    steering_angle = 90
                steer = (9.7/180)*steering_angle + 3
                print(steering_angle)
                p.ChangeDutyCycle(steer)

                key = cv2.waitKey(1)
                if key == 27:
                    break
            else:
                break
               
    else :
        print("can't open video")
finally:
    video.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
