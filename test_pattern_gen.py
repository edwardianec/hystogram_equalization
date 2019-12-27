import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc





#------------------------------MAIN-----------------------------------------


def get_value(i,m,k):
	value = 255

	for j in range(0, (640//m)):
		if (i > j*m and i <= (j+1)*m): value = value-(k*j)
	return value
	


def create_test_pattern():
	height = 480
	width = 640
	first_grade = 255
	img_new = 255 * np.ones(shape=[height, width, 1], dtype=np.uint8)
	for j in range(0,height):
		for i in range(0, width):				
			img_new[j,i] 	= get_value(i,5,2)
			
	return img_new


cv2.imwrite("test_pattern.png",create_test_pattern())
