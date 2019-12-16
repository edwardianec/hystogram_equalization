import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc

import glob



#modified_image = modify_image(img4)
#---------------------------------------------------------

MAX_GREY_LEVEL = 256








def kernel_tree(a, comb, position, variants):	
	for i in comb:
		a[position] = i			
		if (position-1 >= 0):				
			next_position = position -1 
			kernel_tree(a, comb, next_position,variants )
		else:
			variants.append(a[:])	


def build_kernels(comb, kernel_size):
	combinations = []
	a = [0]*(kernel_size*kernel_size)
	list_variants = []
	kernel_tree(a, comb, kernel_size*kernel_size-1, list_variants)
	
	nparray = []
	for variant in list_variants:
		new_arr = np.array( [np.array(variant[i*kernel_size:(i*kernel_size)+kernel_size], dtype="int8") for i in range(0,kernel_size)], dtype="int8")
		nparray.append(new_arr)

	return nparray


def process_image(img_sum_t0, img_sum_t1, image_filename_path, write_path, kernel, kernel_position):
	image_filename 		= image_filename_path.split("\\")[-1]

	img 				= cv2.imread(image_filename_path,  cv2.IMREAD_GRAYSCALE)


	conv_image	 		= cv2.filter2D(img,-1,kernel)
	height, width 		= conv_image.shape	
	img_sum				= 0

	img_sum 			= cv2.sumElems(conv_image)
	#print("conv_sum ", img_sum[0] )
	#print("{0}:{1}".format(kernel_position, [str(i) for i in kernel]))
	if (img_sum[0] == img_sum_t1[0]):
		print("{0}:{1}".format(kernel_position, [str(i) for i in kernel]))
		cv2.imwrite(write_path+str(kernel_position)+".tif", conv_image)

	



#------------------------------MAIN-----------------------------------------

#images 		= glob.glob("D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\*.tif")
def main():

	image_filename_path		= "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\_MG_9778.TIF"
	target_image_0			= "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\_MG_9778_t0.TIF"
	target_image_1			= "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\_MG_9778_t1.TIF"

	t_img0 				= cv2.imread(target_image_0,  cv2.IMREAD_GRAYSCALE)
	t_img1 				= cv2.imread(target_image_1,  cv2.IMREAD_GRAYSCALE)


	img_sum_t0 			= cv2.sumElems(t_img0)
	img_sum_t1 			= cv2.sumElems(t_img1)
	print(img_sum_t0[0], "or ", img_sum_t1[0] )
	
	write_path				= "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\processed\\"
	kernel_position 			= 0
	kernels 				= build_kernels(comb=[-1,-3, 0,1,2,5],kernel_size=3)
	print("total variants: ", len(kernels))	
	

	for kernel in kernels:
		
		process_image(img_sum_t0, img_sum_t1, image_filename_path=image_filename_path, write_path=write_path, kernel=kernel, kernel_position=kernel_position)
		kernel_position = kernel_position + 1
		


#concat_images = np.concatenate( (my_resize(img4,50), my_resize(img4, 50)), axis=1)
#cv2.imshow('modified_image',concat_images )
#my_subplots(img4, img4)


main()






cv2.waitKey(0)
cv2.destroyAllWindows()
gc.collect()
plt.close()

