import csv
import os
from PIL import Image
import cv2
import numpy as np
import os.path

script_dir = os.path.dirname(__file__)
folder_dir = os.path.join(script_dir, 'first_collection')
read_path = os.path.join(folder_dir, 'metadata.csv')
write_path = os.path.join(folder_dir, 'metadata2.csv')

#change parameter if creating new metadata file
create_file = False
image_name = 2
#if we dont create a file,
if(not create_file):
	#edit it 
	csvinput = open(write_path, 'r')
	reader = csv.reader(csvinput)
	num_columns = 0
	omit_array = []
	all_text = []
	#get all omit images
	for i, row in enumerate(reader):
		all_text.append(row)
		num_columns += 1
		image_name = row[1]
		if(row[383] == 'omit'):
			pass
		else:
			if(int(row[383]) == 0):
				omit_array.append(i-1)
#print(num_columns)
if(image_name == 2):
	csvinput = open(read_path, 'r')
	csvoutput = open(write_path, 'w')
	writer = csv.writer(csvoutput, lineterminator = '\n')
	reader = csv.reader(csvinput)
	all_text = []
	num_columns = 0
	omit_array = []
	#create new elements in our metadata file (omit element)
	for i, row in enumerate(reader):
		num_columns += 1
		image_name = row[1]
		if len(row) == 383:
			row.append(1)
		if(row[383] == 0):
			omit_array.append(i)
		all_text.append(row)
	writer.writerows(all_text)
img_path = os.path.join(script_dir, image_name)

###gui stuff###
k = 0
i = 0
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,20)
topLeftCornerOfText = (1000, 20)
topLeftCornerOfText2 = (1000, 40)
fontScale              = .5
fontColor              = (255,255,255)
fontColor2              = (0,0,255)
lineType               = 1
start_int = -1
end_int = -
#if not escape,
while(k != 27):
	csvinput = open(write_path, 'r')
	reader = csv.reader(csvinput)
	#get all rows from file
	row = [row for idx, row in enumerate(reader) if idx == (i+1)][0]
	image_name = row[1]
	img_path = os.path.join(folder_dir, image_name)
	#show image in open cv
	img = cv2.imread(img_path)
	img = cv2.resize(img, (0,0), None, 2, 2)
	#create bar representation for omitted frames at bottom of page
	for e in range(num_columns):
		if(e >= 1280 and e < 1280*2):
			for o in range(949, 959):
				if(e in omit_array and o > 954):
					img[o, e - 1280] = [0,0,255]
				elif(o > 954):
					img[o, e - 1280] = [0,255,0]
				if(i == e):
					img[o, e - 1280] = [255,0,0]
		elif(e<1280*2):
			for o in range(939, 949):
				if(e in omit_array and o > 944):
					img[o, e] = [0,0,255]
				elif(o > 944):
					img[o, e] = [0,255,0]
				if(i == e):
					img[o, e] = [255,0,0]			
	#add image name
	cv2.putText(img, image_name, 
	    bottomLeftCornerOfText, 
	    font, 
	    fontScale,
	    fontColor,
	    lineType)
	cmd_vel_angular_z = row[368]
	cmd_vel_linear_x = row[367]
	#add command vels
	cv2.putText(img, cmd_vel_angular_z, 
	    topLeftCornerOfText, 
	    font, 
	    fontScale,
	    fontColor2,
	    2)
	cv2.putText(img, cmd_vel_linear_x, 
	    topLeftCornerOfText2, 
	    font, 
	    fontScale,
	    fontColor2,
	    2)
	#add group omission number
	cv2.putText(img, str(start_int), 
	    (1100, 920), 
	    font, 
	    fontScale,
	    (0,255,0),
	    2)
	cv2.putText(img, str(end_int), 
	    (1100, 940), 
	    font, 
	    fontScale,
	    (0,255,0),
	    2)
	cv2.putText(img, str(i), 
	    (1040, 930), 
	    font, 
	    fontScale,
	    (0,255,255),
	    2)
	#add more bars
	if(start_int != -1 and end_int != -1 and start_int < end_int):
		if(start_int >= 1280 and start_int >= 1280 and start_int < 1280*2 and end_int < 1280*2):
			for o in range(949, 954):
				img = cv2.line(img, (start_int-1280, o), (end_int-1280, o), (255, 255, 255), 1)
		if(start_int < 1280 and start_int < 1280):
			for o in range(939, 944):
				img = cv2.line(img, (start_int, o), (end_int, o), (255, 255, 255), 1)
	#create circle indicator for omission
	if(i in omit_array):
		cv2.circle(img,(1000,800), 20, (0,0,255), -1)
	else:
		cv2.circle(img,(1000,800), 20, (0,255,0), -1)
	cv2.imshow('image', img)

	#get key stroke
	k = cv2.waitKey(33)
	if(k != -1):
		#print k
		pass
	if(k == ord('d') and i < num_columns - 2):
		i += 1
	if(k == ord('a') and i > 0):
		i -= 1
	if(k == ord('e') and i < num_columns - 2 - 10):
		i += 10
	if(k == ord('q') and i > 0 + 10):
		i -= 10
	#space bar omits
	if(k == ord(' ')):
		if(i in omit_array):
			omit_array.remove(i)
		else:
			omit_array.append(i)
	#r does group omission
	if(k == ord('r')):
		if(start_int > end_int):
			temp = end_int
			end_int = start_int
			start_int = temp
		for j in range(start_int, end_int+1):
			if(j in omit_array):
				omit_array.remove(j)
			omit_array.append(j)
	#t does group addition
	if(k == ord('t')):
		if(start_int > end_int):
			temp = end_int
			end_int = start_int
			start_int = temp
		for j in range(start_int, end_int+1):
			if(j in omit_array):
				omit_array.remove(j)

	#, and . are group omission keys
	if(k == ord(',')):
			start_int = i
	if(k == ord('.')):
		end_int = i

#re-add all data to metadata2
csvoutput = open(write_path, 'w')
writer = csv.writer(csvoutput, lineterminator = '\n')
for i, row in enumerate(all_text):
	if (i in omit_array):
		print(i, ', ')
		all_text[i+1][383] = 0
	else:
		if(i < num_columns - 1):
			all_text[i+1][383] = 1
	#all_text.append(row)
writer.writerows(all_text)
cv2.destroyAllWindows()

