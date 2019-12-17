import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc

def dark_func(z,a,b):
	if (z<= a): 
		y = 1
	elif (z>a and z<= a+b):
		y = 1 - (z-a) / b
	else:
		y = 0
	return y

def grey_func(z, a,b,c):
	if (z < a and z >= a-b):
		y = 1 - (a-z)/b 
	elif (a<=z and z<= a+c):
		y = 1 - (z-a)/c
	else:
		y = 0
	return y

def highlight_func(z,a,b):
	if (a-b <=z and z <= a):
		y = 1 - (a-z)/b 
	elif (z>= a):
		y = 1
	else:
		y = 0
	return y	

def show_fuzz_func():
	print("############ dark func ##############")
	for i in range(0, 20):
		y = dark_func(z=i,a=7,b=3)
		print("x:{0}, y:{1}".format(i,y))

	print("############ grey func ##############")
	for i in range(0, 20):
		y = grey_func(z=i,a=10,b=7,c=6)
		print("x:{0}, y:{1}".format(i,y))

	print("############ highlight func ##############")
	for i in range(0, 20):
		y = highlight_func(z=i,a=13,b=3)
		print("x:{0}, y:{1}".format(i,y))

def defuzification(z,vd,vg,vb):
	ud = dark_func(z=z,a=3,b=7)
	ug = grey_func(z=z,a=10,b=7,c=6)
	ub = highlight_func(z=z,a=16,b=6)
	#print("x:{0},ud:{1},ug:{2},ub:{3}".format(z,ud,ug,ub))
	a = ud*vd+ug*vg+ub*vb
	b = ud+ug+ub
	v = a/b
	print("x:{0}, | a:{1},b:{2} | v:{3};".format(z, a,b,v))

	return v
#-----------------------------------------






for i in [4,5,6,10,12,14,15,16,17]:
	defuzification(i,0,10,20)