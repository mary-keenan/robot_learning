#!/usr/bin/env python

'''
Non-learning Robot Following Algorithm
'''
#355 correct, 710 total
from __future__ import print_function, division
import cv2          # For image processing
import rospy        # For ROS
import numpy as np  # For math
from view_dataset import get_image_index, proc_img
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import csv
import os.path
script_dir = os.path.dirname(__file__)
class FollowerNode():

	def __init__(self, is_live=False, dataset=None):

		# Constants

		# Get label, priority
		# self.label, self.priority = get_label('personfollow')
		self.thresh_min = 0
		self.thresh_max = 200
		self.k_linear = (0.005, 0, 0.3)
		self.k_angular = (0.01, 0, 0.0)

		self.current_image = None
		self.three_point_average = [(0,0), (0,0), (0,0)]
		self.tracking_point = None
		self.image_size = None
		self.robot_command = None
		self.error_sum = [0, 0]
		self.previous_error = [0, 0]
		self.total = 0
		self.correct = 0

		# OpenCV tuning window
		cv2.namedWindow('controls')
		cv2.createTrackbar('thresh_min', 'controls', 0, 255, self.update_thresholds)
		cv2.createTrackbar('thresh_max', 'controls', 0, 255, self.update_thresholds)

		# ROS node setup
		rospy.init_node('baseline_follower')
		self.robot_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
		rospy.Subscriber('/camera/image_raw', Image, self.camera_cb)
		self.update_rate = rospy.Rate(20)
		self.bridge = CvBridge()

		self.is_live = is_live
		if not self.is_live:
			self.dataset = dataset  # dataset
			self.dataset_index = 0       # dataset index

	def camera_cb(self, data):
		try:
			self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	def get_processed_image(self, is_live):
		# Takes boolean, returns processed image

		img = None

		if is_live:
			# Get ROS topic image
			img = self.current_image
			img = proc_img(img)
		else:
			# Get next image in list
			address = '../data/first_collection/' + get_image_index(self.dataset_index) + '.jpg'
			img = cv2.imread(address, 1)
			img = proc_img(img, False, 'blue')

		# Update image size
		self.image_size = (img.shape[0], img.shape[1])

		return img

	def update_tracking_point(self, im):

		# Canny edge detection
		im2 = cv2.Canny(im,self.thresh_min,self.thresh_max)

		# Get contours, find and draw neato contour
		im3, contours, hierarchy = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		if len(contours) == 0:
			rospy.logwarn_throttle(1, "Can't find a contour")
			self.update_tracking_point_average((self.image_size[0]/2, self.image_size[1]/2))
			cv2.circle(im, self.tracking_point, 5,(255,255,255),-1)
			return im

		print(cv2.boundingRect(contours[0]))
		neato_ass = contours[0]
		# for contour in contours:
		#     if self.neato_filter(contour):
		#         neato_ass = contour
		#         break
		#     else:
		#         rospy.logwarn_throttle(1, "Can't find a neato")
		#         self.update_tracking_point_average((self.image_size[0]/2, self.image_size[1]/2))
		#         cv2.circle(im, self.tracking_point, 5,(255,255,255),-1)
		#         return im

		# Draw contours
		cv2.drawContours(im, contours, -1, (100,100,100), 3)
		cv2.drawContours(im, [neato_ass], 0, (0,0,0), 3)

		# Draw circle on center point
		x, y, w, h = cv2.boundingRect(neato_ass)
		self.update_tracking_point_average((int(x + w/2), int(y + h/2)))
		cv2.circle(im, self.tracking_point, 5,(255,255,255),-1)
		cv2.circle(im, (int(self.image_size[1]/2), int(self.image_size[0]/2)), 5, (125,125,125),-1)
		return im

	def update_tracking_point_average(self, new_pt):
		# Given xy point, appends point to running three point average
		a = self.three_point_average[0]
		b = self.three_point_average[1]
		c = self.three_point_average[2]
		self.tracking_point = (int((a[0] + b[0] + c[0])/3), int((a[1] + b[1] + c[1])/3))
		self.three_point_average = [self.three_point_average[1], self.three_point_average[2], new_pt]

	def update_thresholds(self, temp):
		# Update image thresholds based on controls
		self.thresh_min = cv2.getTrackbarPos('thresh_min', 'controls')
		self.thresh_max = cv2.getTrackbarPos('thresh_max', 'controls')

	def neato_filter(self, contour):
		# Given a contour, return True if contour is neato
		x, y, w, h = cv2.boundingRect(contour)
		aspect_ratio = w/h
		if h < 7:
			return True
		else:
			return False
		pass

	def update_twist_command(self, cmd_pt):
		# Given drive point, update drive commands

		error_x = self.image_size[1]/2 - self.tracking_point[0]
		error_y = self.image_size[0]*3/4 - self.tracking_point[1]

		# Calculate error summation
		self.error_sum[0] = self.error_sum[0] + error_x
		self.error_sum[1] = self.error_sum[1] + error_y

		# Calculates derivative, assumes timestep is 0.1s
		errorslope_x = (error_x - self.previous_error[0])/0.2
		errorslope_y = (error_y - self.previous_error[1])/0.2


		# Create nonlinear relationship for control
		msg_ang = self.k_angular[0] * error_x + self.k_angular[2] * errorslope_x #+ self.k_angular[1] * self.error_sum[0]
		msg_lin = self.k_linear[0] * error_y #+ self.k_linear[1] * self.error_sum[1] + self.k_linear[2] * errorslope_y


		# Threshold for Neato
		if msg_ang >= 0.5:
			msg_ang = 0.5
		elif msg_ang < -0.5:
			msg_ang= -0.5
		if msg_lin >= 0.1:
			msg_lin = 0.1
		elif msg_lin < -0.1:
			msg_lin = -0.1

		# Update twist message
		self.robot_command = Twist(Vector3(msg_lin,0,0), Vector3(0,0,msg_ang))
		print("Angular Error:" + str(error_x))
		print("Linear: " + str(self.robot_command.linear.x))
		print("Angular: " + str(self.robot_command.angular.z))
		print("---------------------------------")

		# Update previous errors
		self.previous_error[0] = error_x
		self.previous_error[1] = error_y

	def run_dataset(self):
		# Main loop function for dataset algorithm testing

		# Try to get image, otherwise assume we reached the last image
		try:
			img = self.get_processed_image(self.is_live)
		except AttributeError as e:
			print("Reached end of dataset at image: " + str(self.dataset_index))
			print("Error: " + str(e))
			return 1

		im2 = self.update_tracking_point(img)
		self.update_twist_command(self.tracking_point)
		self.update_accuracy()

		# Show image, increment image to grab
		cv2.imshow(self.dataset, im2)
		cv2.waitKey(10)

		self.dataset_index += 1
		return 0
	def update_accuracy(self):
		folder_dir = os.path.join(script_dir, '../data/first_collection')
		write_path = os.path.join(folder_dir, 'metadata2.csv')
		csvinput = open(write_path, 'r')
		reader = csv.reader(csvinput)
		row = [row for idx, row in enumerate(reader) if idx == (self.dataset_index+1)][0]
		if(int(row[383]) == 1):
			print('aalsdkalsd')
			self.total += 1
			if(float(row[367]) > 0 and self.robot_command.angular.z >0):
				self.correct += 1
			elif(float(row[367]) < 0 and self.robot_command.angular.z < 0):
				self.correct += 1
			elif(float(row[367]) == 0 and self.robot_command.angular.z == 0):
				self.correct += 1
		image_name = row[1]
		print(image_name, self.dataset_index)
		print('correct:', self.correct, 'total:', self.total)
	
	def run_live(self):
		# Main loop function for live algorithm run

		cv2.imshow('image_raw', self.current_image)
		img = self.get_processed_image(self.is_live) # Apply image processing
		im2 = cv2.flip(img, 0) # Flip Image
		im3 = self.update_tracking_point(im2)
		self.update_twist_command(self.tracking_point)
		self.robot_pub.publish(self.robot_command)
		cv2.imshow('image', im3)
		cv2.waitKey(10)
		return 0

	def run(self):
		# Update loop, choses live or on dataset based on init

		# Waits for images to be published
		if self.is_live:
			while not rospy.is_shutdown():
				#print(self.current_image)
				if (self.current_image is not None):
					break
				self.update_rate.sleep()

		while not rospy.is_shutdown():
			if self.is_live:
				if self.run_live():
					break
			else:
				if self.run_dataset():
					break

			self.update_rate.sleep()

if __name__ == "__main__":
	follower = FollowerNode(False)
	follower.run()