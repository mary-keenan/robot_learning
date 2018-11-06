#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
import cv2
import csv
import numpy as np
from scipy.misc import imread
import glob
import random
from sklearn.ensemble import AdaBoostRegressor


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

	def get_data(self):
		""" gets images and velocities from metadata.csv and converts the images to vectors; stores values in x/y_train/test"""
		# import from .npz files
		#----------------------------------------------------------------------
		rawdata = np.load('/home/siena/catkin_ws/src/robot_learning/robot_learning/data_processing_utilities/data/train2.npz')
		x_rawraw = rawdata["imgs"]
		y_raw = rawdata["ang_vels"]
		print('loaded')

		# reshape data
		#----------------------------------------------------------------------
		x_raw = x_rawraw.reshape((x_rawraw.shape[0],
                                               x_rawraw.shape[1]*
                                               x_rawraw.shape[2]*
                                               x_rawraw.shape[3]))
		print('formatted')

		# divide data into train and test
		#----------------------------------------------------------------------
		total_dataset_size = len(x_raw)
		training_set_size = int(round(.5*total_dataset_size * self.proportion_of_training_set_to_total))
		testing_set_size = total_dataset_size - 20# training_set_size
		random.seed(123)
		random.shuffle(x_raw)
		random.shuffle(y_raw)
		self.x_train = x_raw[:500]
		self.y_train = y_raw[:500]
		self.x_test = x_raw[testing_set_size:]
		self.y_test = y_raw[testing_set_size:]


	def run_model(self):
		"""Runs AdaBoostRegressor on training set and compares to test"""
		model = AdaBoostRegressor(n_estimators=2)
		model.fit(self.x_train, self.y_train)
		print('model finished')

	def run(self):
		self.get_data()
		self.run_model()



class ImageVelocityData(object):
	def __init__(self, image_vector, linear_velocity, angular_velocity):
		self.image_vector = image_vector
		self.linear_velocity = linear_velocity
		self.angular_velocity = angular_velocity



if __name__ == '__main__':
    node = AdaBoostReg()
    node.run()
