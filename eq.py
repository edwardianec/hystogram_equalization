import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
	equalization_image	= cv2.equalizeHist(img)


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

def get_hist(img):
	height, width = img.shape
	hist = dict(enumerate([0]*256))
	for j in range(0,height):
		for i in range(0, width):			
			pixel_val = img[j,i]
			hist[pixel_val]	= hist[pixel_val]+1
	return hist	

def cumm_sum():
	pass


def modify_image(img):
	height, width = img.shape
	resolution = height*width
	histogram = get_hist(img)
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
	hystogram = get_hist(img3)
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

concat_images = np.concatenate( (my_resize(img4,50), my_resize(img4, 50)), axis=1)
cv2.imshow('modified_image',concat_images )
my_subplots(img4, img4)


cv2.waitKey(0)
cv2.destroyAllWindows()

