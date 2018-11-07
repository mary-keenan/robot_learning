#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
import cv2
import csv
import numpy as np
from scipy.misc import imread
from skimage.transform import resize
import glob
import random
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import roc_auc_score


class AdaBoostReg(object):
	def __init__(self):
		rospy.init_node('sheepie')
		self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
		self.rate = rospy.Rate(100)
		self.vel_msg = Twist()
		self.vel_msg.linear.x = 0
		self.vel_msg.angular.z = 0
		self.proportion_of_training_set_to_total = .9 # determines what % of the total data will be used for training (with the rest being used for testing)
		self.x_train = []
		self.y_train = []
		self.x_test = []
		self.y_train = []
		self.scoring = 0
		self.most_recent_image = None
		self.predict_img = []

	def get_data(self):
		""" gets images and velocities from metadata.csv and converts the images to vectors; stores values in x/y_train/test"""
		# import from .npz files
		#----------------------------------------------------------------------
		rawdata = np.load('/home/siena/catkin_ws/src/robot_learning/robot_learning/data_processing_utilities/data/train2.npz')
		x_rawraw = rawdata["imgs"]
		y_raw = rawdata["ang_vels"]
		# reshape data
		#----------------------------------------------------------------------
		x_raw = x_rawraw.reshape((x_rawraw.shape[0],
                                               x_rawraw.shape[1]*
                                               x_rawraw.shape[2]*
                                               x_rawraw.shape[3]))
		# divide data into train and test
		#----------------------------------------------------------------------
		total_dataset_size = len(x_raw)
		training_set_size = int(round(.1*total_dataset_size * self.proportion_of_training_set_to_total))
		testing_set_size = total_dataset_size - 20# training_set_size
		random.seed(123)
		random.shuffle(x_raw)
		random.shuffle(y_raw)
		self.x_train = x_raw[:training_set_size]#500]
		self.y_train = y_raw[:training_set_size]#500]
		self.x_test = x_raw[testing_set_size:]
		self.y_test = y_raw[testing_set_size:]

	def run_model(self):
		"""Runs AdaBoostRegressor on training set and compares to test"""
		self.model = AdaBoostRegressor(n_estimators=1)
		self.model.fit(self.x_train, self.y_train)
		#self.scoring = roc_auc_score(self.y_train, model.predict(self.x_train))

	def update_current_image(self, data):
		""" camera callback -- just saves image as most recent image """
		self.most_recent_image = numpy.fromstring(data.data, numpy.uint8)

	def process_img(self, rawimg):
		"""Processes images for prediction"""
		print(type(rawimg))
		rawimg = np.asarray(rawimg)
		print(rawimg.shape)

		img = rawimg[round(rawimg.shape[0]/2):]
		img = resize(img,(120,320))
		self.predict_img = img

	def predict_velocity(self):
		""" makes prediction of the velocities the robot should use to follow another robot given the trained model and input image """
		reshaped_image = self.process_img(self.most_recent_image)
		predicted_encoding = self.model.predict(reshaped_image)    #[0] necessary because the predict() produces a nested list
		linear, angular = [velocities for (velocities, category) in self.encoded_velocities.items() if category == numpy.argmax(predicted_encoding)][0]
		self.vel_msg.linear.x, self.vel_msg.angular.z = (linear * 1.5, angular * 1.5)
		print ('predicted linear velocity: %f, predicted angular velocity: %f' % (self.vel_msg.linear.x, self.vel_msg.angular.z))


	def join_the_herd(self):
		""" neato uses the trained model to navigate (follow another neato) """
		# load model/velocity encodings and set up camera callback
		#self.trained_model = pickle.load(open('/home/siena/catkin_ws/src/robot_learning/robot_learning/data_processing_utilities/data/trained_model_with_omission_84.sav', 'rb'))
		#self.encoded_velocities = pickle.load(open('/home/siena/catkin_ws/src/robot_learning/robot_learning/data_processing_utilities/data/encoded_velocities.sav', 'rb'))
		rospy.Subscriber('camera/image_raw', Image, self.update_current_image)
		# wait for first image data before starting the general run loop
		while self.most_recent_image is None and not rospy.is_shutdown():
			self.rate.sleep()

		# general run loop
		while not rospy.is_shutdown():
			self.predict_velocity()
			self.publisher.publish(self.vel_msg)
			self.rate.sleep()


	def run(self):
		while not rospy.is_shutdown():
			self.get_data()
			print('data')
			self.run_model()
			print('model')
			self.join_the_herd()



class ImageVelocityData(object):
	def __init__(self, image_vector, linear_velocity, angular_velocity):
		self.image_vector = image_vector
		self.linear_velocity = linear_velocity
		self.angular_velocity = angular_velocity



if __name__ == '__main__':
    node = AdaBoostReg()
    node.run()
