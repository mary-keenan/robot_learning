#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
import cv2
import csv
import numpy
#import matplotlib.pyplot as plt
from scipy.misc import imread
import glob
from random import shuffle




class ConvolutionalNeuralNetwork(object):
	def __init__(self):
		rospy.init_node('sheepie')
		self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
		self.rate = rospy.Rate(100)
		self.vel_msg = Twist()
		self.vel_msg.linear.x = 0
		self.vel_msg.angular.z = 0
		self.proportion_of_training_set_to_total = .9 # determines what % of the total data will be used for training (with the rest being used for testing)


	def get_data(self):
		""" gets images and velocities from metadata.csv and converts the images to vectors; returns a list of ImageVelocityData objects """
		# just for first_collection data right now
		
		data_list = []

		# convert metadata.csv to dictionary
		with open('/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/first_collection/metadata.csv') as f:
			rows = [row.split(',') for row in f]

			# figure out the indices of the relevant labels
			labels_list = rows[0]
			image_file_name_index = labels_list.index('image_file_name')
			linear_velocity_index = labels_list.index('cmd_vel_linear_x')
			angular_velocity_index = labels_list.index('cmd_vel_angular_z')

			# organize relevant data in dictionary
			number_of_skipped_rows = 0
			for row in rows[1:100]:
				# convert image name to vector
				image_file_name = row[image_file_name_index]
				try:
					image_file = imread('/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/first_collection/' + image_file_name)
					image_matrix = numpy.asarray(image_file) # 480 x 640 x 3
					image_vector = image_matrix.reshape((1, image_matrix.shape[0] * image_matrix.shape[1] * image_matrix.shape[2]))
					linear_velocity = float(row[linear_velocity_index])
					angular_velocity = float(row[angular_velocity_index])
					if linear_velocity < 10 and angular_velocity < 10: # filter out weird inf values for linear and angular velocity
						data_list.append(ImageVelocityData(image_vector, linear_velocity, angular_velocity))
				except:
					number_of_skipped_rows += 1
					print ("invalid row #%d -- there probably were missing folders in the folder" % number_of_skipped_rows)
					continue

		#print (data_list[10].image_vector, data_list[10].linear_velocity, data_list[10].angular_velocity)
		return data_list


	def divide_training_and_testing_data(self, data_list):
		""" randomly divides total dataset into disjoint training and testing subsets """

		# figure out the sizes of the training and testing datasets
		total_dataset_size = len(data_list)
		training_set_size = int(round(total_dataset_size * self.proportion_of_training_set_to_total))
		testing_set_size = total_dataset_size - training_set_size

		# randomly allocate data points to the two sets
		shuffle(data_list)
		training_set = data_list[:training_set_size]
		testing_set = data_list[training_set_size:]

		return (training_set, testing_set)


	def run(self):
		""" """

		data_list = self.get_data()
		training_list, testing_list = self.divide_training_and_testing_data(data_list)




class ImageVelocityData(object):
	def __init__(self, image_vector, linear_velocity, angular_velocity):
		self.image_vector = image_vector
		self.linear_velocity = linear_velocity
		self.angular_velocity = angular_velocity



if __name__ == '__main__':
    node = ConvolutionalNeuralNetwork()
    node.run()

