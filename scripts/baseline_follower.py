#!/usr/bin/env python


from __future__ import print_function, division
from tensorflow.keras.models import *
from tensorflow.keras.initializers import *
from cv_bridge import CvBridge, CvBridgeError
import cv2
from geometry_msgs.msg import Twist, Vector3
from view_dataset import proc_img
from sensor_msgs.msg import Image
import numpy as np
import rospkg
import rospy
import keras



class Controller(object):
	def __init__(self):
		rospy.init_node("follower")
		rospy.on_shutdown(self.shutdown_func)
		r = rospkg.RosPack()

		#model_path = r.get_path('robot_learning/models/conv_model_1.h5')
		#model_path = r.get_path('robot_learning') + \
        #    "/models/conv_model_1.h5"
		#print(model_path)
		#self.model = load_model(model_path)
		self.model = keras.models.load_model('/home/gretchen/catkin_ws/src/robot_learning/models/conv_model_1.h5')
		#print(self.model.summary())
		self.update_rate = rospy.Rate(10)
		self.motor_ranges = [-0.3, 0.3]
		self.motor_state = [0,0]
		self.curr_image = None
		self.drive_msg = None
		self.bridge = CvBridge()

		self.motor_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
		rospy.Subscriber('/camera/image_raw', Image, self.camera_process)

	def shutdown_func(self):
		#publish robot motors to stop
		self.drive_msg = Twist(linear = Vector3(x=0), angular = Vector3(z=0))
		self.motor_pub.publish(self.drive_msg)

	def camera_process(self, msg):
		#save msg as self.curr_image
		#make black and white
		#cut down image size
		#do something with Non/inf/etc data values
		cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
		self.curr_image = proc_img(cv_image)

	def robot_drive(self, motor_cmds):
		#publish left wheel at self.motor_state[0]
		#publish right wheel at self.motor_state[1]
		ang = motor_cmds[0][0]
		lin = motor_cmds[0][1]
		self.drive_msg = Twist(linear = Vector3(x=lin), angular = Vector3(z=ang))

	def run(self):
		while not rospy.is_shutdown():
		#listen for camera image -> does through subscriber
		#filter data
		#do the neural_net on that data
		#send motor commands (self.robot_drive)

			if self.curr_image is not None:
				img_array = np.expand_dims(self.curr_image, axis =0)
				img_array = np.expand_dims(img_array, axis =3)

				#print(type(self.curr_image))
				vels = self.model.predict(img_array) #rework to follow our actual model's function name
				print(vels)
				self.robot_drive(vels)
				self.motor_pub.publish(self.drive_msg)
		self.update_rate.sleep()

if __name__ == '__main__':
    follower = Controller()
    follower.run()