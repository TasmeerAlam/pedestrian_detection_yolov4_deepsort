# pedestrian_detection_yolov4_deepsort

## Pedestrian Detection and Tracking System


## Project Description

This project develops a Computer Vision Pipeline to automate the analysis of video data for detecting and tracking moving vehicles and pedestrians, utilizing GPS location data and counting systems. The system is designed to meet the needs of urban traffic management and safety enhancement through real-time monitoring and analysis.

## Features

Real-Time Detection and Tracking: Uses state-of-the-art object detection (YOLOv4) and tracking (Deep SORT) algorithms.

Region of Interest (ROI) Analysis: Custom detection of pedestrian crosswalks and traffic signals using OpenCV, enabling precise behavior analysis of pedestrians.

Traffic Signal Status Detection: Analyzes traffic light statuses using OpenCV’s HSV color space to inform pedestrian behavior analysis.

Jaywalking Detection: Combines multiple data points to determine if pedestrians are jaywalking, based on traffic signal status and their location relative to pedestrian crossings.

Direction and GPS Mapping: Assigns markers to objects to determine their direction across intersections and maps pixel coordinates to GPS coordinates using perspective transformation.


## Technology Stack

OpenCV: For image processing and video analysis.

YOLOv4: For real-time object detection.

Deep SORT: For object tracking.
Python: Primary programming language.

Usage
The system processes video input to detect and track pedestrians and vehicles, outputting the tracking results and relevant GPS data. Users can input custom video files and specify parameters for detection and tracking within the script.

