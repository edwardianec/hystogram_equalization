import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc

import glob

img0 = cv2.imread('images/src/0.tif',0)
img1 = cv2.imread('images/src/1.tif',0)
img2 = cv2.imread('images/src/2.tif',0)
img3 = cv2.imread('images/src/3.tif',0)
img4 = cv2.imread('images/src/4.tif',0)


def filters(img):

	#kernel 		= np.ones((5,5),np.float32)/25
	kernel_edges 		= [
		[0,	-2,	 0],
		[-2, 0,	 2],
		[0,	 2,	 0]
	]
	kernel_clarity 		= [
		[-1,	-1,	 -1],
		[-1, 	9,	 -1],
		[-1,	-1,	 -1]
	]

	kernel_clarity_0 		= [
		[0,	-1,	 0],
		[-1, 5,	 -1],
		[0,	-1,	 0]
	]

	kernel_gausian 		= [
		[1/16,	2/16,	 1/16],
		[2/16, 4/16,	 2/16],
		[1/16,	2/16,	 1/16]
	]

	kernel_edges_mat 			= np.array(kernel_edges, dtype=float);
	kernel_clarity_mat 			= np.array(kernel_clarity, dtype=float);
	kernel_clarity_0_mat 		= np.array(kernel_clarity_0, dtype=float);
	kernel_gausian_mat 			= np.array(kernel_gausian, dtype=float);

	edges_image 		= cv2.filter2D(img,-1,kernel_edges_mat)
	clarity_image 		= cv2.filter2D(img,-1,kernel_clarity_mat)
	clarity_0_image 	= cv2.filter2D(img,-1,kernel_clarity_0_mat)
	gausian_image 		= cv2.filter2D(img,-1,kernel_gausian_mat)


	bilateral_image		= cv2.bilateralFilter(img,9,5,5)
	clahe 				= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image 		= clahe.apply(gausian_image)

#
#
#bioinspired
#
#
#
def get_retina(cv_img):
	#bm3d = cv2.imread('images/src/coffeebm3dsharp.tif',0)
	retina = cv2.bioinspired.Retina_create((cv_img.shape[1], cv_img.shape[0]))
	
	# the retina object is created with default parameters. If you want to read
	# the parameters from an external XML file, uncomment the next line
	retina.setup('setup.xml')
	
	# feed the retina with several frames, in order to reach 'steady' state
	retina.run(cv_img)

	# get our processed image :)
	return	retina.getParvo()


def modify_image_core(val):
	new_val = 0

	if (val >= 127): new_val = val*2
	else: new_val = val

	new_val = 255 if new_val>255 else new_val
	return new_val



def cumm_sum():
	pass


def modify_image(img):
	height, width = img.shape
	resolution = height*width
	histogram = build_hyst(img)
	mult = 255/resolution
	new_greys = {}
	for i in range(0,256):
		new_greys[i] = cumm_sum(i)*mult


	#img_new = np.zeros((height,width))
	img_new = 255 * np.ones(shape=[height, width, 1], dtype=np.uint8)
	for j in range(0,height):
		for i in range(0, width):
			pixel_val = img[j,i]			
			new_val = modify_image_core(pixel_val)	
			img_new[j,i] = new_val
	return img_new


def my_resize(img, percent):
	width = int(img.shape[1] * percent / 100)
	height = int(img.shape[0] * percent / 100)
	dim = (width, height)
	# resize image
	return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)





def my_subplots(img1, img2):
	hist, edges = np.histogram(img1,bins=range(255))
	hist2, edges2 = np.histogram(img2,bins=range(255))
	hystogram = build_hist(img3)
	lag = 0.1
	x = list(range(1,256))
	y = [(hystogram[out]+hystogram[out-1]) for out in x if out>0 ]

	ax = plt.subplot(2, 2, 1) 

	# Draw the plot
	ax.bar(edges[:-1], hist, width = 0.5, color='#0504aa')
	# Title and labels
	ax.set_title('Histogram of image 1', size = 10)
	ax.set_xlabel('grey level', size = 10)
	ax.set_ylabel('amount', size= 10)

	ax = plt.subplot(2, 2, 2)    
	# Draw the plot
	ax.bar(edges2[:-1], hist2, width = 0.5, color='#0504aa') 
	# Title and labels
	ax.set_title('Histogram of image 2', size = 10)
	ax.set_xlabel('grey level', size = 10)
	ax.set_ylabel('amount', size= 10)

	ax = plt.subplot(2, 2, 3) 
	ax.plot(x,y)

	plt.tight_layout()
	plt.show()


#modified_image = modify_image(img4)
#---------------------------------------------------------

MAX_GREY_LEVEL = 256

def build_hist(img):
	height, width = img.shape
	hist = dict(enumerate([0]*256))
	for j in range(0,height):
		for i in range(0, width):			
			pixel_val = int(img[j,i])
			#print("pix_val:",pixel_val)
			hist[pixel_val]	= hist[pixel_val]+1

	return hist	

def normilize_image(img, histogram):
	height, width = img.shape
	hist_minimum = 0
	hist_maximum = 0
	for grey in range(0,256):		
		if histogram[grey]:
			if (hist_minimum == 0):	hist_minimum = grey
			hist_maximum = grey

	stretch_coefficient = 255/(hist_maximum)
	normilized_image = np.zeros(shape=[height, width], dtype=np.uint8)
	
	for i in range(0,width):
		for j in range(0, height):
			pixel_val = img[j,i]			
			new_val = int((pixel_val-hist_minimum)*stretch_coefficient)			
			normilized_image[j,i] = new_val #if new_val <=255 else 255
			
	min_percent = (hist_minimum / (width*height))*100
	max_percent = (hist_maximum / (width*height))*100
	#print("HIST_MINIMUM:{0}, HIST_MAXIMUM:{1} ;".format(hist_minimum, hist_maximum))
	#print("HIST_MINIMUM %:{0}, HIST_MAXIMUM %:{1} ;".format(min_percent, max_percent))
	return normilized_image

def cumul_hist_val(histogram, grey):
	#print(histogram)
	cum = 0
	for i in range(0, grey):
		
		cum = cum + histogram[i]
	return cum

def cdf_trashold(histogram, grey, trashold):
	cum = 0
	for i in range(0, grey):
		hist_val = histogram[i] if histogram[i] < trashold else trashold
		cum = cum + hist_val
	return cum


def cumulative_function(histogram):
	cum_dict = {}
	for i in range(0,256):
		cum_dict[i] =  cumul_hist_val(histogram,i)
	
	return (cum_dict.keys(), cum_dict.values())





def equalization_image(img, histogram):
	height, width = img.shape	
	eq_image = np.zeros(shape=[height, width], dtype=np.uint8)
	eq_image.fill(0)

	for i in range(0,width):
		for j in range(0, height):
			pixel_val = img[j,i]			
			new_val = 	(255/(width*height))*cumul_hist_val(histogram, pixel_val)	
			eq_image[j,i] = int(new_val)

	return eq_image

# adaptive trashold
def equalization_image_2(img, histogram, trashold):
	height, width = img.shape	
	eq_image = np.zeros(shape=[height, width], dtype=np.uint8)

	multiplyer = (MAX_GREY_LEVEL-1)/cdf_trashold(histogram, MAX_GREY_LEVEL-1, trashold)

	for i in range(0,width):
		for j in range(0, height):
			pixel_val = img[j,i]			
			new_val = 	multiplyer*cdf_trashold(histogram, pixel_val, trashold)	
			eq_image[j,i] = int(new_val)

	return eq_image



def show_graphs(graphs, width, height, image_filename_path=""):
	figure, axs 		= plt.subplots(height, width,  gridspec_kw={'hspace': 0.5, 'wspace': 0.5})
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
		plt.savefig(image_path+"\\processed\\figures\\"+image_filename+"_figure.png")
	else:
		plt.show()
	figure.clf()
	plt.close()


def show_graphs_simple(graph1, graph2, width):
	figure, (axs1, axs2 ) = plt.subplots( width,  gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

	axs1.bar(graph1[0], graph1[1])
	axs1.set_title("hist 0")

	axs2.bar(graph2[0], graph2[1])
	axs2.set_title("hist 1")
	plt.show()	

def hist_peaks(h):
	last_val_item = 0
	peaks = {}
	peaks[0] = 0
	for i in range(1, MAX_GREY_LEVEL-1):
		if (h[i]):
			last_val_item = i
			if ((h[i] > h[i-1]) and (h[i] > h[i+1])):
				peaks[i] = h[i]
			else: peaks[i] = 0
		else: peaks[i] = 0
	return peaks
			
def find_peaks_avarage(h):
	counter = 0
	sum 	= 0
	for i in range(1, MAX_GREY_LEVEL):
		if (h[i]):
			counter = counter + 1
			sum = sum + h[i]
	avarage = int(sum/counter)
	return avarage


def process_image(image_filename_path):
	#print("image_filename_path:", image_filename_path)
	image_filename 		= image_filename_path.split("\\")[-1]
	image_path			=  "\\".join(image_filename_path.split("\\")[:-1])

	img 				= cv2.imread(image_filename_path, cv2.IMREAD_GRAYSCALE)
	height, width 		= img.shape
	src_hyst 			= build_hist(img)

	normilized_image 	= normilize_image(img, src_hyst)
	norm_hyst			= build_hist(normilized_image)

	#simple equalization
	eq_image 			= equalization_image(img, src_hyst)
	eq_hyst				= build_hist(eq_image)

	#peaks of image
	peaks_src_hist		= hist_peaks(src_hyst)
	peak_trashold		= find_peaks_avarage(src_hyst)
	#print("trashold:",peak_trashold)

	#equalization with trashold
	eq_trashold_img		= equalization_image_2(img, src_hyst, peak_trashold/5)
	eq_trashold_hyst	= build_hist(eq_trashold_img)



	graph_src_hyst				= [src_hyst.keys(), src_hyst.values()]
	graph_eq_hyst				= [eq_hyst.keys(), eq_hyst.values()]
	graph_norm_hyst				= [norm_hyst.keys(), norm_hyst.values()]
	graph_eq_trashold_hyst		= [eq_trashold_hyst.keys(), eq_trashold_hyst.values()]
	graph_peaks_hyst			= [peaks_src_hist.keys(), peaks_src_hist.values()]




	show_graphs( [
		graph_src_hyst, 
		graph_norm_hyst ,
		graph_eq_hyst,
		graph_eq_trashold_hyst
		#graph_peaks_hyst,
		

		#cumulative_function(src_hyst),
		#cumulative_function(norm_hyst),
		#cumulative_function(eq_hyst),
		#cumulative_function(eq_trashold_hyst),
		#cumulative_function(peaks_src_hist)

		
	], width=2, height=2, image_filename_path=image_filename_path)

	row0 = np.concatenate( (my_resize(img,80), my_resize(eq_image, 80)), axis=0)
	row1 = np.concatenate( (my_resize(normilized_image,80), my_resize(eq_trashold_img, 80)), axis=0)
	full_image = np.concatenate( (row0, row1), axis=1)
	#cv2.imshow('modified_image', full_image)
	#print("write_path:", image_writepath+"\\"+image_filename)
	cv2.imwrite(image_path+"\\processed\\images\\"+image_filename, full_image)

	#del a, b
	#gc.collect()
	#cv2.imshow('modified_image',eq_image)




#------------------------------MAIN-----------------------------------------
src_path = "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\video\M11-2231\\tiff\\*.tif"

images 		= glob.glob(src_path)
img_amount 	= len(images)
img_counter = 0
for image in images:
	img_counter = img_counter + 1
	print("{0} of {1}".format(img_counter, img_amount))
	process_image(image)

#concat_images = np.concatenate( (my_resize(img4,50), my_resize(img4, 50)), axis=1)
#cv2.imshow('modified_image',concat_images )
#my_subplots(img4, img4)

gc.collect()
cv2.waitKey(0)
cv2.destroyAllWindows()

