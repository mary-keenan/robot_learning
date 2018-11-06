#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
import cv2
import csv
import numpy
import pickle
from scipy.misc import imread
import glob
from random import shuffle
from keras import layers, models, activations
from keras.utils import to_categorical
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
from vis.visualization import visualize_activation, visualize_saliency
from vis.utils import utils
from PIL import Image as pil_Image


class ConvolutionalNeuralNetwork(object):
	def __init__(self):
		rospy.init_node('sheepie')

		# training
		self.proportion_of_training_set_to_total = .8 # determines what % of the total data will be used for training (with the rest being used for testing)
		
		# publishers and subscribers
		self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
		self.saliency_pub = rospy.Publisher('saliency', Image, queue_size = 10)
		self.rate = rospy.Rate(20)
		self.vel_msg = Twist()
		self.vel_msg.linear.x = 0
		self.vel_msg.angular.z = 0
		self.saliency_msg = Image()
		self.saliency_msg.header.stamp = rospy.Time.now()
		self.saliency_msg.encoding = 'rgb8'

		# global variables
		self.most_recent_image = None
		self.trained_model = None
		self.encoded_velocities = None

# ------------------------train model-------------------------

	def get_data(self, num_pictures = None, folders_to_omit = []):
		""" gets images and velocities from metadata.csv and converts the images to vectors; returns a list of ImageVelocityData objects """		
		data_list = []
		number_of_skipped_rows = 0
		number_of_omitted_rows = 0

		for folder in glob.glob('/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/*collection/'):

			if folder in folders_to_omit:
				continue

			# add the metadata.csv's to a common dictionary
			with open(folder + 'metadata2.csv') as f:
				rows = [row.split(',') for row in f]

				# figure out the indices of the relevant labels
				labels_list = rows[0]
				image_file_name_index = labels_list.index('image_file_name')
				linear_velocity_index = labels_list.index('cmd_vel_linear_x')
				angular_velocity_index = labels_list.index('cmd_vel_angular_z')
				omit_entry_index = 383

				# organize relevant data in dictionary
				for row in rows[1:]:
					if num_pictures is not None and len(data_list) >= num_pictures:
						return data_list

					# convert image name to vector
					image_file_name = row[image_file_name_index]
					try:
						omit_entry = row[omit_entry_index]
						if omit_entry == 1:
							number_of_omitted_rows += 1
							print ("omitted row #%d" % number_of_omitted_rows)
							continue
						image_file = imread(folder + image_file_name)
						image_vector = numpy.asarray(image_file)[220:420, 0::4, :] # 200 x 160 x  x 3
						linear_velocity = float(row[linear_velocity_index])
						angular_velocity = float(row[angular_velocity_index])
						if linear_velocity < 10 and angular_velocity < 10: # filter out weird inf values for linear and angular velocity
							data_list.append(ImageVelocityData(image_vector, linear_velocity, angular_velocity))
					except:
						number_of_skipped_rows += 1
						print ("invalid row #%d -- there probably were missing folders in the folder" % number_of_skipped_rows)
						continue
		
		return data_list


	def double_data(self, data_list):
		""" for photos with an angular velocity, add a flipped duplicate to the list of data """
		doubled_data_list = []

		for data_point in data_list:
			doubled_data_list.append(data_point)
			if data_point.angular_velocity > 0 or data_point.angular_velocity < 0:
				flipped_data_point = ImageVelocityData(numpy.flip(data_point.image_vector, 0), data_point.linear_velocity, - data_point.angular_velocity)
				doubled_data_list.append(flipped_data_point)

		print (len(doubled_data_list))
		return doubled_data_list


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


	def separate_x_and_y(self, data_list):
		""" deconstructs ImageVelocityData objects to x and y components for testing """
		images_list = []
		velocities_list = []

		for image_velocity_data in data_list:
			images_list.append(image_velocity_data.image_vector)
			velocities_list.append((image_velocity_data.linear_velocity, image_velocity_data.angular_velocity))

		images_list = numpy.asarray(images_list).astype('float32')/255
		return (images_list, velocities_list)


	def train_model(self, folders_to_omit = [], should_save = True):
		""" trains a CNN model with preprocessed data """

		# prepare data for model
		data_list = self.get_data(folders_to_omit)
		doubled_data_list = self.double_data(data_list)
		training_list, testing_list = self.divide_training_and_testing_data(doubled_data_list)
		training_x_list, training_y_list = self.separate_x_and_y(training_list)
		testing_x_list, testing_y_list = self.separate_x_and_y(testing_list)

		# convert velocities to categories -- they're actually discrete but they look continuous
		velocity_category_values = set(training_y_list) # use unique values as categories
		velocity_category_dict = dict() # maps velocity value to category #
		category_number = 0

		for category_value in velocity_category_values:
			velocity_category_dict[category_value] = category_number
			category_number += 1
		
		training_y_categorized = [velocity_category_dict[velocity_value] for velocity_value in training_y_list]
		testing_y_categorized = [velocity_category_dict[velocity_value] for velocity_value in testing_y_list]
		
		for key, val in velocity_category_dict.iteritems():
			print (key, training_y_categorized.count(val))

		training_y_hot_encoded = to_categorical(training_y_categorized, len(velocity_category_values))
		testing_y_hot_encoded = to_categorical(testing_y_categorized, len(velocity_category_values))

		# build model
		model = models.Sequential() # allows you to add multiple layers?
		model.add(layers.Conv2D(2, (5, 5), activation ='relu', input_shape = (200, 160, 3))) # 32 filters, 5x5 weight matrix
		model.add(layers.MaxPooling2D(2,2)) # 2x2 window for finding the best layers (highest weights) and only keeping them?
		model.add(layers.Conv2D(4, (5, 5), activation = 'relu'))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Flatten()) # converts 3D tensor to 1D tensor for the softmax layer
		model.add(layers.Dense(len(velocity_category_values), activation = 'softmax')) # final layer
		#model.summary()

		# train model
		model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
		model.fit(training_x_list, training_y_hot_encoded, validation_data=(testing_x_list, testing_y_hot_encoded), batch_size = 10, epochs = 1, verbose = 1)

		# save model + encoding dictionary unless otherwise specified (ie for training) to disk
		if should_save:
			pickle.dump(model, open('trained_model_test.sav', 'wb'))
			pickle.dump(velocity_category_dict, open('encoded_velocities.sav', 'wb'))

		return (model, velocity_category_dict)

# ----------------------visualize model-----------------------

	def visualize_model(self):
		""" visualizes model layers and saliency maps for troubleshooting """

		# load model
		model = pickle.load(open('/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/trained_model_with_omission_84.sav', 'rb'))
		model.summary()
		data_list = self.get_data(805)
		testing_x_list, testing_y_list = self.separate_x_and_y(data_list)
		name = 'omission_84_2'
		pic_num = 800

		# NOTE: make sure to change this to the last layer of the model
		layer_idx = utils.find_layer_idx(model, 'dense_1')

		plt.rcParams['figure.figsize'] = (18, 6)

		# swap softmax with linear (this is needed when you have a softmax at the output layer)
		model.layers[layer_idx].activation = activations.linear
		model = utils.apply_modifications(model)

		img = visualize_activation(model, layer_idx)
		plt.imshow(img)
		plt.gca().grid(False)
		model.predict(img[numpy.newaxis,:,:,:])
		plt.savefig("/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/visualized_layers/" + name + "_dense_layer.png")
		plt.show()

		grads = visualize_saliency(model, layer_idx, filter_indices = 0, seed_input = testing_x_list[pic_num])
		# Plot with 'jet' colormap to visualize as a heatmap.
		plt.subplot(1,2,1)
		plt.imshow(testing_x_list[pic_num], cmap='jet')
		plt.gca().grid(False)
		plt.subplot(1,2,2)
		plt.imshow(grads, cmap='jet')
		plt.gca().grid(False)

		plt.savefig("/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/visualized_layers/" + name + "_saliency_map.png")
		plt.show()

# -------------------------test model-------------------------

	def test_model(self):
		""" trains model with only some of the data and uses the first_collection dataset to test the accuracy of the model """

		# train model
		# model, encoded_velocities = self.train_model(
		# 	folders_to_omit = ["/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/first_collection/"], 
		# 	should_save = True)

		model = pickle.load(open('/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/trained_model_test.sav', 'rb'))
		encoded_velocities = pickle.load(open('/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/encoded_velocities.sav', 'rb'))

		# prepare testing data
		testing_data = self.get_data(
			folders_to_omit = ["/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/second_collection/", 
							   "/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/first_collection/"])
		testing_x_list, testing_y_list = self.separate_x_and_y(testing_data)
		categorized_testing_y_list = [encoded_velocities[velocities] for velocities in testing_y_list]
		encoded_testing_y_list = to_categorical(categorized_testing_y_list, 5)

		# test model
		test_loss, test_acc = model.evaluate(testing_x_list, encoded_testing_y_list)
		print('Test accuracy:', test_acc) # 88.03%

# -------------------------run robot--------------------------
	
	def update_current_image(self, data):
		""" camera callback -- just saves image as most recent image """
		self.most_recent_image = numpy.fromstring(data.data, numpy.uint8)


	def predict_velocity(self):
		""" makes prediction of the velocities the robot should use to follow another robot given the trained model and input image """
		reshaped_image = self.most_recent_image.reshape(1, 480, 640, 3)[:, 220:420, 0::3, :]
		predicted_encoding = self.trained_model.predict(reshaped_image)[0] # [0] necessary because the predict() produces a nested list
		linear, angular = [velocities for (velocities, category) in self.encoded_velocities.items() if category == numpy.argmax(predicted_encoding)][0]
		self.vel_msg.linear.x, self.vel_msg.angular.z = (linear * 1.5, angular * 1.5)
		print ('predicted linear velocity: %f, predicted angular velocity: %f' % (self.vel_msg.linear.x, self.vel_msg.angular.z))

		# layer_idx = utils.find_layer_idx(self.trained_model, 'dense_1')
		# grads = visualize_saliency(self.trained_model, layer_idx, filter_indices = 0, seed_input = reshaped_image)
		# # Plot with 'jet' colormap to visualize as a heatmap.
		# # plt.subplot(1,2,1)
		# # plt.imshow(reshaped_image.reshape(200, 214, 3), cmap='jet')
		# # plt.gca().grid(False)
		# # plt.subplot(1,2,2)
		# # plt.imshow(grads, cmap='jet')
		# # plt.gca().grid(False)
		# # plt.savefig("/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/visualized_layers/saliency_map_live.png")
		# print (grads[0,0])
		# self.saliency_msg.data = numpy.array(cv2.cvtColor(grads, cv2.COLOR_BGR2GRAY)).tostring()


	def join_the_herd(self):
		""" neato uses the trained model to navigate (follow another neato) """

		# load model/velocity encodings and set up camera callback
		self.trained_model = pickle.load(open('/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/trained_model_with_omission_84.sav', 'rb'))
		self.encoded_velocities = pickle.load(open('/home/mary/catkin_ws/src/robot_learning/data_processing_utilities/data/encoded_velocities.sav', 'rb'))

		rospy.Subscriber('camera/image_raw', Image, self.update_current_image)

		# wait for first image data before starting the general run loop
		while self.most_recent_image is None and not rospy.is_shutdown():
			self.rate.sleep()

		# general run loop
		while not rospy.is_shutdown():
			self.predict_velocity()
			self.publisher.publish(self.vel_msg)
			# self.saliency_pub.publish(self.saliency_msg)
			self.rate.sleep()


class ImageVelocityData(object):
	def __init__(self, image_vector, linear_velocity, angular_velocity):
		self.image_vector = image_vector
		self.linear_velocity = linear_velocity
		self.angular_velocity = angular_velocity


if __name__ == '__main__':
	node = ConvolutionalNeuralNetwork()
	# node.train_model()
	# node.visualize_model()
	# node.test_model()
	node.join_the_herd()


