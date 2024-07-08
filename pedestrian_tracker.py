##main code
##Implementing direction
##Implenting GPS coordinates from pixel value
##04/18/2022
### command prompt input: python direction_gps_exp.py --video ./data/video/dc.mp4 --output ./outputs/dc_result.mp4 --model yolov4 --info

import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import pandas as pd 
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import pixel_to_gps 
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from collections import deque ##yehengchen
import warnings


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
					'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.1, 'iou threshold') #original script
flags.DEFINE_float('score', 0.1, 'score threshold') #original script: 0.50
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

global traffic_signal
#add new code for color and pts
pts = [deque(maxlen=300) for _ in range(9999)]
warnings.filterwarnings('ignore')
# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
##end of new code for color

def main(_argv):
	# Definition of the parameters
	max_cosine_distance = 0.4
	nn_budget = None
	nms_max_overlap = 1.0
	
	# initialize deep sort
	model_filename = 'model_data/mars-small128.pb'
	encoder = gdet.create_box_encoder(model_filename, batch_size=1)
	# calculate cosine distance metric
	metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	# initialize tracker
	tracker = Tracker(metric)

	# load configuration for object detector
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)
	STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
	input_size = FLAGS.size
	video_path = FLAGS.video

	# load tflite model if flag is set
	if FLAGS.framework == 'tflite':
		interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		print(input_details)
		print(output_details)
	# otherwise load standard tensorflow saved model
	else:
		saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
		infer = saved_model_loaded.signatures['serving_default']

	# begin video capture
	try:
		vid = cv2.VideoCapture(int(video_path))
	except:
		vid = cv2.VideoCapture(video_path)

	out = None

	# get video ready to save locally if flag is set
	if FLAGS.output:
		# by default VideoCapture returns float instead of int
		width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(vid.get(cv2.CAP_PROP_FPS))
		codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
		out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

	rows = []
	frame_num = 0
	first_frame = True

	# while video is running
	while True:
		return_value, frame1 = vid.read()
		if return_value:
			frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
			image = Image.fromarray(frame1)
		else:
			print('Video has ended or failed, try a different video format!')
			break
		frame_num +=1
		df1 = pd.DataFrame({'Frame': [frame_num]})
		print('Frame #: ', frame_num)
		frame_size = frame.shape[:2]
		image_data = cv2.resize(frame, (input_size, input_size))
		image_data = image_data / 255.
		image_data = image_data[np.newaxis, ...].astype(np.float32)
		start_time = time.time()


		if first_frame== True: 
			#Traffic Light selection
			roi_1 = cv2.selectROI(frame1)
			#Pedestrian Crossing down Selection
			roi_2 = cv2.selectROI(frame1)
			#Pedestrian crossing up selection
			roi_3 = cv2.selectROI(frame1)
			#Pedestrian Crossing left Selection
			roi_4 = cv2.selectROI(frame1)
			#Pedestrian Crossing right Selection
			roi_5 = cv2.selectROI(frame1)

			first_frame = False


		#Pedestrian corssing Coordinates (down)
		y1_2 = int(roi_2[1])
		y2_2 = int(roi_2[1]+roi_2[3])
		x1_2 = int(roi_2[0])
		x2_2 = int(roi_2[0]+roi_2[2])

		#Pedestrian corssing Coordinates (up)
		y1_3 = int(roi_3[1])
		y2_3 = int(roi_3[1]+roi_3[3])
		x1_3 = int(roi_3[0])
		x2_3 = int(roi_3[0]+roi_3[2])

		#Pedestrian corssing Coordinates (left)
		y1_4 = int(roi_4[1])
		y2_4 = int(roi_4[1]+roi_4[3])
		x1_4 = int(roi_4[0])
		x2_4 = int(roi_4[0]+roi_4[2])

		
		#Pedestrian corssing Coordinates (right)
		y1_5 = int(roi_5[1])
		y2_5 = int(roi_5[1]+roi_5[3])
		x1_5 = int(roi_5[0])
		x2_5 = int(roi_5[0]+roi_5[2])

		roiColor = frame1[int(roi_1[1]):int(roi_1[1]+roi_1[3]), int(roi_1[0]):int(roi_1[0]+roi_1[2])]
		
		#Traffic Signal status selection
		hsv = cv2.cvtColor(roiColor,cv2.COLOR_BGR2HSV)

		#red
		lower_hsv_red = np.array([157,177,122])
		upper_hsv_red = np.array([179,255,255])
		mask_red = cv2.inRange(hsv,lowerb=lower_hsv_red,upperb=upper_hsv_red)
		#Median filtering
		red_blur = cv2.medianBlur(mask_red, 7)
		#green
		lower_hsv_green = np.array([49,79,137])
		upper_hsv_green = np.array([90,255,255])
		mask_green = cv2.inRange(hsv,lowerb=lower_hsv_green,upperb=upper_hsv_green)
		#Median filtering
		green_blur = cv2.medianBlur(mask_green, 7)

		#Because the image is a binary image, so if the image has a white point, which is 255, then take his maximum value of 255
		red_color = np.max(red_blur)
		green_color = np.max(green_blur)
		#Judging the binary image in red_color if the value is equal to 255, then it is judged as red
		if red_color == 255:
			traffic_signal = "red"
		#Judge the binary image in green_color if the value is equal to 255, then judge it as green
		elif green_color == 255:
			traffic_signal = "green"
		
		# run detections on tflite if flag is set
		if FLAGS.framework == 'tflite':
			interpreter.set_tensor(input_details[0]['index'], image_data)
			interpreter.invoke()
			pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
			# run detections using yolov3 if flag is set
			if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
				boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
												input_shape=tf.constant([input_size, input_size]))
			else:
				boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.1,
												input_shape=tf.constant([input_size, input_size])) #original script: 0.25
		else:
			batch_data = tf.constant(image_data)
			pred_bbox = infer(batch_data)
			for key, value in pred_bbox.items():
				boxes = value[:, :, 0:4]
				pred_conf = value[:, :, 4:]

		boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
			boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
			scores=tf.reshape(
				pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
			max_output_size_per_class=50,
			max_total_size=50,
			iou_threshold=FLAGS.iou,
			score_threshold=FLAGS.score
		)

		# convert data to numpy arrays and slice out unused elements
		num_objects = valid_detections.numpy()[0]
		bboxes = boxes.numpy()[0]
		bboxes = bboxes[0:int(num_objects)]
		scores = scores.numpy()[0]
		scores = scores[0:int(num_objects)]
		classes = classes.numpy()[0]
		classes = classes[0:int(num_objects)]

		# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
		original_h, original_w, _ = frame.shape
		bboxes = utils.format_boxes(bboxes, original_h, original_w)

		# store all predictions in one parameter for simplicity when calling functions
		pred_bbox = [bboxes, scores, classes, num_objects]

		# read in all class names from config
		class_names = utils.read_class_names(cfg.YOLO.CLASSES)

		# by default allow all classes in .names file
		#allowed_classes = list(class_names.values())
		
		# custom allowed classes (uncomment line below to customize tracker for only people)
		allowed_classes = ['person', 'car', 'bicycle', 'motorbike', 'bus', 'truck']

		# loop through objects and use class index to get class name, allow only classes in allowed_classes list
		names = []
		deleted_indx = []
		for i in range(num_objects):
			class_indx = int(classes[i])
			class_name = class_names[class_indx]
			if class_name not in allowed_classes:
				deleted_indx.append(i)
			else:
				names.append(class_name)
		names = np.array(names)
		count = len(names)
		if FLAGS.count:
			cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
			print("Objects being tracked: {}".format(count))
		# delete detections that are not in allowed_classes
		bboxes = np.delete(bboxes, deleted_indx, axis=0)
		scores = np.delete(scores, deleted_indx, axis=0)

		# encode yolo detections and feed to tracker
		features = encoder(frame, bboxes)
		detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

		#initialize color map
		cmap = plt.get_cmap('tab20b')
		colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

		# run non-maxima supression
		boxs = np.array([d.tlwh for d in detections])
		scores = np.array([d.confidence for d in detections])
		classes = np.array([d.class_name for d in detections])
		indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
		detections = [detections[i] for i in indices]       

		# Call the tracker
		tracker.predict()
		tracker.update(detections)

        #####new code#####
        ##from###

		i = int(0)
		indexIDs = []
		c = []
		counter = []
		for track in tracker.tracks:
			if not track.is_confirmed() or track.time_since_update > 1:
				continue
			
			indexIDs.append(int(track.track_id))
			counter.append(int(track.track_id))
			bbox = track.to_tlbr()


			color = colors[int(track.track_id) % len(colors)] 
			color = [i * 255 for i in color] 

			class_name = track.get_class() 

			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
			cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

			i += 1
			
			center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
			center = (int(((bbox[2]-bbox[0])/2.0)+bbox[0]), int(bbox[3]))
			
			pts[track.track_id].append(center)
			thickness = 5
			
			cv2.circle(frame,  (center), 1, color, thickness)

			###Jaywalking condition
			#jaywalking on pedestrian crossing
			center_y = int(((bbox[1])+(bbox[3]))/2)
			center_x = int(((bbox[0])+(bbox[2]))/2)

			#Jaywalking
			if class_name == 'person':
				if center_y >= y1_2 and center_y <= y2_2:
					if center_x >= x1_2 and center_x <= x2_2:
						marker = '1'
						if traffic_signal == 'green':
							jay_status = 'Jaywalking'
						if traffic_signal == 'red':
							jay_status = "Not Jaywalking"
						cv2.putText(frame, jay_status,(int(bbox[0]), int(bbox[1]-30)),0, 0.75, (255,255,255),2)
				elif center_y >= y1_3 and center_y <= y2_3:
					if center_x >= x1_3 and center_x <= x2_3:
						marker = '2'
						if traffic_signal == 'green':
							jay_status = 'Jaywalking'
						if traffic_signal == 'red':
							jay_status = "Not Jaywalking"
						cv2.putText(frame, jay_status,(int(bbox[0]), int(bbox[1]-30)),0, 0.75, (255,255,255),2)
				elif center_y >= y1_4 and center_y <= y2_4:
					if center_x >= x1_4 and center_x <= x2_4:
						marker = '3'
						if traffic_signal == 'green':
							jay_status = 'Not Jaywalking'
						if traffic_signal == 'red':
							jay_status = "Jaywalking"
						cv2.putText(frame, jay_status,(int(bbox[0]), int(bbox[1]-30)),0, 0.75, (255,255,255),2)
				elif center_y >= y1_5 and center_y <= y2_5:
					if center_x >= x1_5 and center_x <= x2_5:
						marker = '4'
						if traffic_signal == 'green':
							jay_status = 'Not Jaywalking'
						if traffic_signal == 'red':
							jay_status = "Jaywalking"
						cv2.putText(frame, jay_status,(int(bbox[0]), int(bbox[1]-30)),0, 0.75, (255,255,255),2)

				else:
					jay_status = "NA"
					marker = '0'
			#Vehicle direction
			#elif class_name == ['car', 'bus', 'motorcycle', 'bicycle', 'truck']:
			else:
				jay_status = "NA"
				if center_y >= y1_2 and center_y <= y2_2:
					if center_x >= x1_2 and center_x <= x2_2:
						marker = '1'
				elif center_y >= y1_3 and center_y <= y2_3:
					if center_x >= x1_3 and center_x <= x2_3:
						marker = '2'
				elif center_x >= x1_4 and center_x <= x2_4:
					if center_x >= x1_4 and center_x <= x2_4:
						marker = '3'
				elif center_x >= x1_5 and center_x <= x2_5:
					if center_x >= x1_5 and center_x <= x2_5:
						marker = '4'
				else:
					marker = '0'



			##draw ROI on video
			#draw and out text for traffic signal
			cv2.rectangle(frame, (int(roi_1[0]),int(roi_1[1])), (int(roi_1[0]+roi_1[2]),int(roi_1[1]+roi_1[3])), (0,0,255), 2)
			cv2.putText(frame, traffic_signal, (int(roi_1[0]), int(roi_1[1]-10)),0, 0.75, (255,255,255),2)


			#pixel to gps transformation
			#pixel coordinates corresponding to the top left, top right, bottom right, bottom left
			quad_coords = {"lonlat": np.array([[ 38.903528, -77.026111], [ 38.903723, -77.026124], [ 38.903621, -77.025859], [ 38.903456, -77.025902]]),"pixel": np.array([[293,292], [513, 293], [527, 392], [47, 329]])} #mass_10th
			#quad_coords = {"lonlat": np.array([[ 38.902515, -77.0367579], [ 38.902703, -77.0366936], [ 38.9026743, -77.0363397], [ 38.9026032, -77.0365342]]),"pixel": np.array([[142,193], [536,227], [553,365], [23,205]])} #k_16th
			pm = pixel_to_gps.PixelMapper(quad_coords["pixel"], quad_coords["lonlat"]) 
			lonlat = pm.pixel_to_lonlat(center)
			lat = float("{:.2f}".format(lonlat[0,0]))
			lon = float("{:.2f}".format(lonlat[0,1]))


        #####end of new code#####

		# if enable info flag then print details about each track
			rows.append([frame_num, time.time(), traffic_signal, str(track.track_id), class_name, jay_status,center[0], center[1], lonlat[0,0], lonlat[0,1], marker])

			if FLAGS.info:
				print("Tracker ID: {}, Class: {}, Center(x,y): {}, GPS(lat, long):{}, Marker: {}".format(str(track.track_id), class_name, (center[0], center[1]), (lonlat[0,0], lonlat[0,1]), marker))

		cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
		# calculate frames per second of running detections
		fps = 1.0 / (time.time() - start_time)
		print("FPS: %.2f" % fps)
		result = np.asarray(frame)
		result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		
		if not FLAGS.dont_show:
			cv2.imshow("Output Video", result)
		
		# if output flag is set, save video file
		if FLAGS.output:
			out.write(result)
		if cv2.waitKey(1) & 0xFF == ord('q'): break
	cv2.destroyAllWindows()

	df=pd.DataFrame(rows, columns = ['Frame', 'Time', 'Traffic Signal Status', 'Tracker ID', 'Class', 'Jaywalking', 'Center(x)', 'Center(y)', 'GPS(lat)', 'GPS(lon)', 'Marker'])
	df.to_csv('mass_10th.csv', encoding = 'utf-8', index = False)

if __name__ == '__main__':
	try:
		app.run(main)
	except SystemExit:
		pass
