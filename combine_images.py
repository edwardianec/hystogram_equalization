import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL

import glob


def combine_images(img_path, write_path):
	image_filename 	= img_path.split("\\")[-1]	
	imgs    = []
	print("src:",image_path)
	print("out:", "src\\vid\\created\\figures\\"+image_filename+"_figure.png")
	imgs.append(PIL.Image.open(image_path).convert('LA'))
	imgs.append(PIL.Image.open("src\\vid\\created\\figures\\"+image_filename+"_figure.png").convert('LA'))
	# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
	min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
	print("min",min_shape)

	imgs_comb = np.hstack( [np.asarray( i.resize(min_shape) ) for i in imgs ] )

	# save that beautiful picture
	imgs_comb = PIL.Image.fromarray( imgs_comb)
	imgs_comb.save( write_path+image_filename+"_combined.png" )    

	# for a vertical stacking it is simple: use vstack
	#imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
	#imgs_comb = PIL.Image.fromarray( imgs_comb)
	#imgs_comb.save( 'Trifecta_vertical.jpg' )


#------------------------------MAIN-----------------------------------------

images 		= glob.glob("src\\vid\\created\\*.tif")
write_path	= "src\\vid\\created\\combined_images\\"
img_amount 	= len(images)
img_counter = 0
for image_path in images:
	img_counter = img_counter + 1
	print("{0} of {1}".format(img_counter, img_amount))
	combine_images(image_path, write_path)
print("DONE!")
#concat_images = np.concatenate( (my_resize(img4,50), my_resize(img4, 50)), axis=1)
#cv2.imshow('modified_image',concat_images )
#my_subplots(img4, img4)


