import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL

import glob


def combine_images(image_path, figures_path, write_path):
	image_filename 	= image_path.split("\\")[-1]	
	imgs    = []
	
	
	imgs.append(PIL.Image.open(image_path).convert('LA'))
	imgs.append(PIL.Image.open(figures_path+image_filename+"_figure.png").convert('LA'))
	# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
	min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
	

	imgs_comb = np.hstack( [np.asarray( i.resize(min_shape) ) for i in imgs ] )

	# save that beautiful picture
	imgs_comb = PIL.Image.fromarray( imgs_comb)
	imgs_comb.save( write_path+image_filename+"_combined.png" )    

	# for a vertical stacking it is simple: use vstack
	#imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
	#imgs_comb = PIL.Image.fromarray( imgs_comb)
	#imgs_comb.save( 'Trifecta_vertical.jpg' )


#------------------------------MAIN-----------------------------------------

images 		= glob.glob("D:\\resilio\\ip_lib_ed\\src_images\\dslr\\video\\M11-2231\\tiff\\processed\\images\\*.tif")
figures		= "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\video\\M11-2231\\tiff\\processed\\figures\\"
write_path	= "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\video\\M11-2231\\tiff\\processed\\combined\\"

img_amount 	= len(images)
img_counter = 0
for image_path in images:
	img_counter = img_counter + 1
	print("{0} of {1}".format(img_counter, img_amount))
	combine_images(image_path=image_path, figures_path=figures, write_path=write_path)
print("DONE!")
#concat_images = np.concatenate( (my_resize(img4,50), my_resize(img4, 50)), axis=1)
#cv2.imshow('modified_image',concat_images )
#my_subplots(img4, img4)


