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
		new_arr = np.array( [np.array(variant[i*kernel_size:(i*kernel_size)+kernel_size], dtype="uint8") for i in range(0,kernel_size)], dtype="uint8")
		nparray.append(new_arr)

	return nparray


def process_image(image_filename_path, write_path, kernel, kernel_position):
	image_filename 		= image_filename_path.split("\\")[-1]

	img 				= cv2.imread(image_filename_path,  cv2.IMREAD_GRAYSCALE)
	height, width 		= img.shape	

	conv_image	 		= cv2.filter2D(img,-1,kernel)

	cv2.imwrite(write_path+str(kernel_position)+".tif", conv_image)



#------------------------------MAIN-----------------------------------------

#images 		= glob.glob("D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\*.tif")
def main():

	image_filename_path		= "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\_MG_9778.TIF"
	write_path				= "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\processed\\"
	kernel_position 			= 0
	kernels 				= build_kernels(comb=[0,1,2],kernel_size=3)
	print("total variants: ", len(kernels))	

	for kernel in kernels:
		print("{0}:{1}".format(kernel_position, [str(i) for i in kernel]))
		process_image(image_filename_path=image_filename_path, write_path=write_path, kernel=kernel, kernel_position=kernel_position)
		kernel_position = kernel_position + 1


#concat_images = np.concatenate( (my_resize(img4,50), my_resize(img4, 50)), axis=1)
#cv2.imshow('modified_image',concat_images )
#my_subplots(img4, img4)


main()






cv2.waitKey(0)
cv2.destroyAllWindows()
gc.collect()
plt.close()

