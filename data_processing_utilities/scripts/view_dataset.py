#!/usr/ebin/env python

from __future__ import print_function, division #THE FUTURE!!!!!
import cv2          # For image processing
import sys          # For command line arguments
import numpy as np  # Math.
import os.path      # Used for checking existence of files

FLIP = False
SCALE = 30
COLOR = 'full'

def run():

    flag_view = False
    flag_save = False
    dataset = None
    filename = None

    # If one arg, view dataset
    if len(sys.argv) == 2:
        dataset = sys.argv[1]
        flag_view == True
        print("Viewing dataset " + dataset)

    # If two args, view dataset, save new data file
    elif len(sys.argv) == 3:
        dataset = sys.argv[1]
        filename = sys.argv[2]
        flag_view = flag_save = True
        print("Saving dataset " + dataset + " to " + filename)

    # Otherwise, screw you
    else:
        print("ERR: Wrong number of arguments")
        return

    #data_array =np.zeros((2822, 106, 192, 3))
    data_array = np.zeros((get_dataset_size(dataset) + 1, 106, 192, 3), dtype=np.uint8)
    image_index = 0
    image = None

    # Loop through images
    while True:

        # Gets address of image in dataset
        address = get_image_address(dataset, image_index)
        print("Processing " + address)

        # Kill everything . . . if image does not exist
        if not os.path.isfile(address):
            print("Reached end of dataset")
            cv2.destroyAllWindows()
            break

        # read, process, show, and save the image
        image = cv2.imread(address, 1)
        image_processed = proc_img(image, FLIP, COLOR)  # Process image
        cv2.imshow(dataset, image_processed)      # Show image

        cv2.waitKey(10)

        # If we're saving images, store current image in data_array
        if flag_save:
            data_array[image_index,:,:,:] = image_processed


        image_index += 1

    if flag_save:
        print("Saving file to " + filename)
        np.save(filename, data_array)

def proc_img(img, flip=True, color='gray'):
    top_cut_range = 125

    if flip:
        img = cv2.flip(img, 0)

    # Crop top section of image
    height, width, channels = img.shape
    img_crop = img[top_cut_range:height, 0:width]
    if color == 'gray':
        img_color = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    elif color == 'red':
        img_color = img_crop[:,:,0]
    elif color == 'green':
        img_color = img_crop[:,:,1]
    elif color == 'blue':
        img_color = img_crop[:,:,2]
    else:
        img_color = img_crop

    #img_norm = cv2.equalizeHist(img_color)
    img_dwnsmpl = rescale_frame(img_color, SCALE)
    img_blur = cv2.GaussianBlur(img_dwnsmpl,(9,9),0)
    return img_blur

def rescale_frame(frame, percent=25):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    ret_frame = cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    return ret_frame

def get_image_index(num):
    index_len = 10      # Number of digits in file index

    if 0 <= num < 10:
        return "000000000" + str(num)
    elif 1 <= num / 10 < 10:
        return "00000000" + str(num)
    elif 1 <= num / 100 < 10:
        return "0000000" + str(num)
    elif 1 <= num / 1000 < 10:
        return "000000" + str(num)
    elif 1 <= num / 10000 < 10:
        return "00000" + str(num)
    elif 1 <= num / 100000 < 10:
        return "0000" + str(num)
    elif 1 <= num / 1000000 < 10:
        return "000" + str(num)
    elif 1 <= num / 10000000 < 10:
        return "00" + str(num)
    elif 1 <= num / 100000000 < 10:
        return "0" + str(num)
    else:
        return str(num)

def get_image_address(dataset, index):
    # Given dataset and image index, returns address of image
    return '../data_processing_utilities/data/' + dataset + '/' + get_image_index(index) + '.jpg'

def get_dataset_size(dataset):
    # Given folder location, gets highest indexed image in folder
    index = 0
    while(True):
        if os.path.isfile(get_image_address(dataset, index)):
            index += 1
        else:
            return index - 1


if __name__ == '__main__':
    run()