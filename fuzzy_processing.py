import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc
import glob
import statistics

#-------------------------------------------------------------------------------------------------------------
#	Операции с гистограммой

def build_hist(img):
	height, width = img.shape
	hist = dict(enumerate([0]*256))
	for j in range(0,height):
		for i in range(0, width):			
			pixel_val = int(img[j,i])
			#print("pix_val:",pixel_val)
			hist[pixel_val]	= hist[pixel_val]+1

	return hist	

def build_first_derivative(hist):
	max = 255
	derivative = dict(enumerate([0]*(max+1)))
	j = 0
	for i in range(0, max+1):
		hist_val = hist[i]
		if (i > 0 and i <max):
			derivative[j] = (hist[i+1] - hist[i-1])/2		
		j = j +1

	return derivative

def build_second_derivative(hist):
	max = 255
	derivative = dict(enumerate([0]*(max+1)))
	j = 0
	for i in range(0, max+1):
		hist_val = hist[i]
		if (i > 0 and i <max):
			derivative[j] = hist[i+1] - 2*hist[i] + hist[i-1]		
		j = j +1

	return derivative

def filtered_hist(h):
	max = 255
	filtered = dict(enumerate([0]*(max+1)))
	for i in range(0, max+1):		
		if (i==0 or i==max): filtered[i] == 0
		else: filtered[i] = statistics.median([h[i-1],h[i], h[i+1]])
	return filtered	



def find_local_max(h):
	max = 255
	local_max = dict(enumerate([0]*(max+1)))
	j = 0
	for i in range(0, max+1):		
		if (i > 0 and i < max):
			if (h[i+1]==0 and  h[i-1]==0):
				local_max[j] = 0
			elif ((h[i-1] == h[i+1] and h[i] >= h[i+1] ) or (h[i-1] < h[i] and h[i+1] < h[i])):
				local_max[j] = h[i]
			else:
				local_max[j] = 0
		j = j +1	
	return local_max

def find_regions(h):
	max = 255
	right_index = 0
	left_index = 0
	for i in range(max,-1,-1):
		if (h[i]>0): 
			right_index = i;
			break			
	for i in range(0,(max+1)):
		if (h[i]>0): 
			left_index = i;
			break	
	step = int((right_index-left_index)/2)
	
	#med1 = statistics.median([h[i] for i in range(left_index+1, left_index+step)])
	med1 = statistics.median([h[i] for i in range(left_index, left_index+step) if (h[i] > 0)])
	med2 = statistics.median([h[i] for i in range(left_index+step, right_index) if (h[i] > 0)])

	medpos1 = [i for i in range(left_index+1, left_index+step) if (h[i] == med1)]
	medpos2 = [i for i in range(left_index+step, right_index) if (h[i] == med2)]

	return (left_index, right_index, (med1, medpos1), (med2, medpos2))

def cumul_hist_val(histogram, grey):
	#print(histogram)
	cum = 0
	for i in range(0, grey):
		
		cum = cum + histogram[i]
	return cum

def cumulative_function(histogram):
	cum_dict = {}
	for i in range(0,256):
		cum_dict[i] =  cumul_hist_val(histogram,i)
	
	return (cum_dict.keys(), cum_dict.values())


#-------------------------------------------------------------------------------------------------------------
#	Функции принадлежности к множествам
#	и функция вывода множест в виде графиков

def sigma_left_func(z,a,b, max_top=1):
	if (z > a and z < b):
		y = (1 - (z-a)/(b-a) )*max_top
	elif (z<= a):
		y = 1*max_top
	else:
		y = 0
	return y	

def trianglural_func(z, a,b,c, max_top=1):
	if (z < b and z >= a):
		y = (1 - (b-z)/(b-a) )*max_top
	elif (z >= b and z < c):
		y = (1 - (z-b)/(c-b) )*max_top
	else:
		y = 0
	return y


def sigma_right_func(z,a,b, max_top=1):
	if (z > a and z <= b):
		y = (1 - (b-z)/(b-a) )*max_top
	elif (z>= b):
		y = 1*max_top
	else:
		y = 0
	return y	

def plt_member_functions(params, member_vds, max_top=1):
	a,b,c,d,e				= params
	dark_func_dots			= [a,b] # трапеция
	grey_func_dots			= [a,b,c] # треугольная функция
	grey_mid_func_dots		= [b,c,d] # треугольная функция
	grey_right_func_dots	= [c,d,e] # треугольная функция	
	white_func_dots			= [d, e] #трапеция

	y_0		= [sigma_left_func(x,a,b, max_top=max_top) for x in range(0,256)]
	y_1 	= [trianglural_func(x,a,b,c, max_top=max_top) for x in range(0,256)]	
	y_2 	= [trianglural_func(x,b,c,d, max_top=max_top) for x in range(0,256)]	
	y_3 	= [trianglural_func(x,c,d,e, max_top=max_top) for x in range(0,256)]	
	y_4 	= [sigma_right_func(x,d,e, max_top=max_top) for x in range(0,256)]	

	y_5		= [x*15 for x in range(0,255)]
	x_5		= [member_vds[0]]*255
	y_6		= [x*15 for x in range(0,255)]
	x_6		= [member_vds[1]]*255
	y_7		= [x*15 for x in range(0,255)]
	x_7		= [member_vds[2]]*255
	y_8		= [x*15 for x in range(0,255)]
	x_8		= [member_vds[3]]*255
	y_9		= [x*15 for x in range(0,255)]
	x_9		= [member_vds[4]]*255



	x 		= range(0,256)	
	
	return plt.plot(x, y_0, x, y_1, x, y_2, x, y_3, x, y_4, x_5, y_5,  x_6, y_6,  x_7, y_7,  x_8, y_8,  x_9, y_9)


#-------------------------------------------------------------------------------------------------------------

# 	описание формулы на странице 233, Мир цифровой обработки 
def defuzification(ums, vds ):
	(a, b) = 0, 0
	for i in range(0,len(ums)):
		a = a + ums[i]*vds[i]
		b = b + ums[i]
	v = a/b
	#print("x:{0}, | a:{1},b:{2} | v:{3};".format(z, a,b,v))

	return int(v)

def get_defuzzification_list(params, vds):
	a,b,c,d,e		= params
	
	defuzz_list = []
	for z in range(0, 256):
		ums 			= []
		ums.append(sigma_left_func(z, a,b))
		ums.append(trianglural_func(z, a,b,c))
		ums.append(trianglural_func(z, b,c,d))
		ums.append(trianglural_func(z, c,d,e))
		ums.append(sigma_right_func(z, d,e))		
		defuzz_list.append(defuzification(ums,vds))
	return  defuzz_list



#-------------------------------------------------------------------------------------------------------------

# 	графики
#  


def show_cdf_func(h):
	graph = cumulative_function(h)
	
	plt.plot(list(graph[0]),list(graph[1]) )
	plt.show()	

def show_cdf_derivative(h):
	cdf_graph 	= cumulative_function(h)
	cdf_dict	= dict(zip(cdf_graph[0], cdf_graph[1]))
	deriv_graph = build_first_derivative(cdf_dict)
	plt.plot(list(deriv_graph.keys()),list(deriv_graph.values()) )
	plt.show()		
	



def show_transformation_func(member_params, member_vds ):
	plt.plot(list(range(0,256)), get_defuzzification_list(member_params, member_vds), list(range(0,256)), list(range(0,256)))
	plt.show()


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
	plt.close()


def show_graph(graph, member_params, member_vds, image_filename_path=""):
	max = 255
	#plt.figure(figsize=(30, 20))
	#plt.cla()
	#plt.clf()
	plt.tick_params(axis='both', which='major', labelsize=5)
	plt.tick_params(axis='both', which='minor', labelsize=8)
	bar_tiks = []
	for i in range(0,max+1):
		val = list(graph[1])[i]
		if (val > 0):
			bar_tiks.append(i)
		else:		
			bar_tiks.append("")

	plt.bar(graph[0], graph[1], tick_label=bar_tiks, )

	plt_member_functions(member_params, member_vds, max_top=35000)
	
	
	image_filename 		= image_filename_path.split("\\")[-1]
	image_path			=  "\\".join(image_filename_path.split("\\")[:-1])
	#plt.xticks(range(0, 256), range(0,256), rotation=90)
	for i in range(0,max+1):
		val = list(graph[1])[i]
		if (val > 0):
			plt.text(x = i , y = val, s = val, ha='center', va='bottom', size = 5)

	if (image_filename_path):
		plt.savefig(image_path+"\\processed\\"+image_filename+"_figure.png")
	else:		
		plt.show()

	

#-----------------------------------------

def get_fuzzy_image(img, params, vds):
	height, width 	= img.shape	
	fuzzy_image 	= np.zeros(shape=[height, width], dtype=np.uint8)
	fuzzy_list		= get_defuzzification_list(params, vds)
	for i in range(0,width):
		for j in range(0, height):
			pixel_val = img[j,i]			
			new_val = 	fuzzy_list[pixel_val]	
			fuzzy_image[j,i] = new_val
	return fuzzy_image


def fuzzy_process(path, wrt_path):
	image_filename 		= path.split("\\")[-1]
	image_path			=  "\\".join(path.split("\\")[:-1])

	img 				= cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	height, width 		= img.shape

	member_params		= (0,30,80,229,255)
	member_vds			= [0,63,126, 188, 255]

	fuzzy_image 		= get_fuzzy_image(img, member_params, member_vds)

	src_hyst 			= build_hist(img)	
	fuzz_hyst 			= build_hist(fuzzy_image)

	filtered_h			= filtered_hist(src_hyst)
	filtered_fuzzy		= filtered_hist(fuzz_hyst)

	
	graph_src_hyst		= [src_hyst.keys(), src_hyst.values()]
	graph_fuzz_hyst		= [fuzz_hyst.keys(), fuzz_hyst.values()]

	graph_filtered_hyst = [filtered_h.keys(), filtered_h.values()]
	graph_fuzz_filtered_hyst = [filtered_fuzzy.keys(), filtered_fuzzy.values()]


	src_scnd_deriv		= build_second_derivative(src_hyst)
	local_max			= find_local_max(src_hyst)
	local_filt_max		= find_local_max(filtered_h)


	graph_src_maxs		= [local_max.keys(), local_max.values()]
	graph_filt_maxs		= [local_filt_max.keys(), local_filt_max.values()]
	graph_sec_der		= [src_scnd_deriv.keys(), src_scnd_deriv.values()]



	""" 	show_graphs( [
			graph_src_hyst, 
			graph_filtered_hyst,
			graph_src_maxs, 
			graph_filt_maxs		
		], width=2, height=2, image_filename_path=path) """
	print(find_regions(local_max))
	
	
	#show_graph(graph_sec_der)

	cv2.imwrite(wrt_path+image_filename, fuzzy_image)
	
	#show_cdf_derivative(src_hyst)
	
	show_cdf_func(src_hyst)	
	show_graph(graph_src_maxs, member_params, member_vds)
	show_transformation_func(member_params, member_vds)
	#show_graph(graph_src_hyst, member_params, member_vds)
	show_graph(graph_fuzz_hyst, member_params, member_vds)

	
	#


#------------------------------MAIN-----------------------------------------

src_path = "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\fuzzy\\src\\*.tif"
wrt_path = "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\fuzzy\\processed\\"

images 		= glob.glob(src_path)
img_amount 	= len(images)
img_counter = 0
for image in images:
	img_counter = img_counter + 1
	#print("{0} of {1}".format(img_counter, img_amount))
	fuzzy_process(image, wrt_path)

#concat_images = np.concatenate( (my_resize(img4,50), my_resize(img4, 50)), axis=1)
#cv2.imshow('modified_image',concat_images )
#my_subplots(img4, img4)

gc.collect()
cv2.waitKey(0)
cv2.destroyAllWindows()

