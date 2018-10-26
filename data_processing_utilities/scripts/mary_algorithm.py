#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
import cv2
import csv
import numpy
import matplotlib.pyplot as plt
from scipy.misc import imread
import glob




class ConvolutionalNeuralNetwork(object):
	def __init__(self):
		rospy.init_node('sheepie')
		self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
		self.rate = rospy.Rate(100)
		self.vel_msg = Twist()
		self.vel_msg.linear.x = 0
		self.vel_msg.angular.z = 0


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



	def run(self):
		self.get_data()




class ImageVelocityData(object):
	def __init__(self, image_vector, linear_velocity, angular_velocity):
		self.image_vector = image_vector
		self.linear_velocity = linear_velocity
		self.angular_velocity = angular_velocity



if __name__ == '__main__':
    node = ConvolutionalNeuralNetwork()
    node.run()

