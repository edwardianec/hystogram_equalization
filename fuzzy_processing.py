import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc
import glob



def build_hist(img):
	height, width = img.shape
	hist = dict(enumerate([0]*256))
	for j in range(0,height):
		for i in range(0, width):			
			pixel_val = int(img[j,i])
			#print("pix_val:",pixel_val)
			hist[pixel_val]	= hist[pixel_val]+1

	return hist	

def build_derivative(hist):
	derivative = dict(enumerate([0]*256))
	j = 0
	for i in range(0, 256):
		hist_val = hist[i]
		if (hist_val > 0):
			if (i > 0): derivative[j] = hist[i] - hist[i-1]
			else: derivative[j] = 0
			j = j +1

	return derivative



#------------------------------------
#	Функции принадлежности к 4м  множествам

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

def grey_right_func(z, a,b,c):
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

#----------------------

def defuzification(z,vd,vg,vrg, vb, dark_params, grey_params,grey_right_param,  white_params ):
	ud = dark_func(z=z,a=dark_params["a"],b=dark_params["b"])
	ug = grey_func(z=z,a=grey_params["a"],b=grey_params["b"],c=grey_params["c"])
	ugr = grey_right_func(z=z,a=grey_right_param["a"],b=grey_right_param["b"],c=grey_right_param["c"])
	ub = highlight_func(z=z,a=white_params["a"],b=white_params["b"])
	#print("x:{0},ud:{1},ug:{2},ub:{3}".format(z,ud,ug,ub))
	a = ud*vd+ug*vg+ugr*vrg+ub*vb
	b = ud+ug+ugr+ub
	v = a/b
	print("x:{0}, | a:{1},b:{2} | v:{3};".format(z, a,b,v))

	return int(v)

def get_defuzzification_list():
	(vd,vg,vrg, vb)			= (0,100,150, 255)
	dark_func_dots			= [0,30] # трапеция
	grey_func_dots			= [0,30,190] # треугольная функция
	grey_right_func_dots	= [30,190,255] # треугольная функция	
	white_func_dots			= [190, 255] #трапеция

	dark_params 		= {"a":dark_func_dots[0], "b":dark_func_dots[1]-dark_func_dots[0]}
	grey_params 		= {"a":grey_func_dots[1], "b":grey_func_dots[1]-grey_func_dots[0], "c":grey_func_dots[2]-grey_func_dots[1]}
	grey_right_param 	= {"a":grey_right_func_dots[1], "b":grey_right_func_dots[1]-grey_right_func_dots[0], "c":grey_right_func_dots[2]-grey_right_func_dots[1]}
	white_params 		= {"a":white_func_dots[1], "b":white_func_dots[1]-white_func_dots[0]}

	defuzz_list = []
	for i in range(0, 256):
		defuzz_list.append(defuzification(i,vd,vg,vrg, vb, dark_params, grey_params,grey_right_param, white_params))

	return  defuzz_list


def show_graphs(graphs, width, height, image_filename_path=""):
	figure, axs 		= plt.subplots(height, width,  gridspec_kw={'hspace': 0.5, 'wspace': 0.5}, figsize=(30,20))
	image_filename 		= image_filename_path.split("\\")[-1]
	image_path			=  "\\".join(image_filename_path.split("\\")[:-1])

	i,j = 0,0
	for graph in graphs:		
		axs[j,i].bar(graph[0], graph[1])
		axs[j,i].set_title("graph {0}, {1}".format(j,i))
		i = i+1
		if (i>width-1):
			i = 0
			j = j+1
	if (image_filename_path):
		plt.savefig(image_path+"\\processed\\"+image_filename+"_figure.png")
	else:
		plt.show()
	figure.clf()
	plt.close()


#-----------------------------------------



def get_fuzzy_image(img):
	height, width 	= img.shape	
	fuzzy_image 	= np.zeros(shape=[height, width], dtype=np.uint8)
	fuzzy_list		= get_defuzzification_list()
	for i in range(0,width):
		for j in range(0, height):
			pixel_val = img[j,i]			
			new_val = 	fuzzy_list[pixel_val]	
			fuzzy_image[j,i] = new_val
	return fuzzy_image



def fuzzy_process(path):
	image_filename 		= path.split("\\")[-1]
	image_path			=  "\\".join(path.split("\\")[:-1])

	img 				= cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	height, width 		= img.shape

	fuzzy_image 		= get_fuzzy_image(img)

	src_hyst 			= build_hist(img)
	fuzz_hyst 			= build_hist(fuzzy_image)

	
	graph_src_hyst		= [src_hyst.keys(), src_hyst.values()]
	graph_fuzz_hyst		= [fuzz_hyst.keys(), fuzz_hyst.values()]

	src_deriv			= build_derivative(src_hyst)
	fuzz_deriv			= build_derivative(fuzz_hyst)

	graph_src_deriv		= [src_deriv.keys(), src_deriv.values()]
	graph_fuz_deriv		= [fuzz_deriv.keys(), fuzz_deriv.values()]

	show_graphs( [
		graph_src_hyst, 
		graph_fuzz_hyst,
		graph_src_deriv, 
		graph_fuz_deriv		
	], width=2, height=2, image_filename_path=path)

	cv2.imwrite(image_path+"\\processed\\"+image_filename, fuzzy_image)


#------------------------------MAIN-----------------------------------------

src_path = "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\fuzzy\\*.tif"


images 		= glob.glob(src_path)
img_amount 	= len(images)
img_counter = 0
for image in images:
	img_counter = img_counter + 1
	print("{0} of {1}".format(img_counter, img_amount))
	fuzzy_process(image)

#concat_images = np.concatenate( (my_resize(img4,50), my_resize(img4, 50)), axis=1)
#cv2.imshow('modified_image',concat_images )
#my_subplots(img4, img4)

gc.collect()
cv2.waitKey(0)
cv2.destroyAllWindows()

