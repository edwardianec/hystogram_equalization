import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2, itertools
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

def find_equal_regions(h):
	sum_pix = cumul_hist_val(h, 255)
	region = sum_pix//7
	print("pixel_counter: {0}; region: {1}".format(sum_pix, region))
	cum = 0
	j = 0
	region_grade = []
	for i in range(0, 256):
		cum = cum + h[i]
		if (cum > region): 
			cum = 0
			region_grade.append((i+1))
			
			if (j == 5):
				break
			j = j+1
	for i in range(0, len(region_grade)):
		if (i+1==len(region_grade)):
			avarage = get_region_avarage(region_grade[i], 255)
		else: avarage = get_region_avarage(region_grade[i], region_grade[i+1])
	return region_grade


def maximums(graph, maximums_count):
	# данный метод позволяет найти максимумы на графике
	# и выбрать из него только необходимое количество из
	# всех максимумов. При этом максимумы выбираются начиная
	# с самых больших значений. 
	maximums = {}
	previous = 0
	for i in range(1, len(graph)):
		current = graph[i] - graph[i-1]
		if (current < 0 and previous > 0): maximums[i-1] = graph[i-1]
		previous = current

	# После того, как мы нашли максимумы, мы должны упорядичить эти
	# максимумы в порядке убывания, после чего выбрать только необходимое нам количество,
	# определенное в переменной maximums_count
	maximums	= {k: v for k, v in sorted(maximums.items(), key=lambda item: item[1], reverse=True)}
	maximums 	= dict(itertools.islice(maximums.items(),maximums_count))
	maximums	= dict(sorted(maximums.items()))
	return maximums

#-------------------------------------------------------------------------------------------------------------
#	Функции принадлежности к множествам
#	и функция вывода множест в виде графиков

def sigma_left_func(z,params, max_top=1):
	a,b = params[0], params[1]
	if (z > a and z < b):
		y = (1 - (z-a)/(b-a) )*max_top
	elif (z<= a):
		y = 1*max_top
	else:
		y = 0
	return y	

def trianglural_func(z, params, max_top=1):
	a,b,c = params[0], params[1], params[2]	
	if (z < b and z >= a):
		y = (1 - (b-z)/(b-a) )*max_top
	elif (z >= b and z < c):
		y = (1 - (z-b)/(c-b) )*max_top
	else:
		y = 0
	return y


def sigma_right_func(z,params, max_top=1):
	a,b = params[0], params[1]	
	if (z > a and z <= b):
		y = (1 - (b-z)/(b-a) )*max_top
	elif (z>= b):
		y = 1*max_top
	else:
		y = 0
	return y	

def plt_member_functions(params, member_vds, max_top=1):
	max_grade 	= 256
	y 			= []
	#member_params		= [0,30,60,80,100,150,229,255]
	y.append( [sigma_left_func(x, params[0:2], max_top=max_top) for x in range(0,max_grade)]		)
	y.append( [trianglural_func(x,params[0:3], max_top=max_top) for x in range(0,max_grade)]		)
	y.append( [trianglural_func(x,params[1:4], max_top=max_top) for x in range(0,max_grade)]		)
	y.append( [trianglural_func(x,params[2:5], max_top=max_top) for x in range(0,max_grade)]		)
	y.append( [trianglural_func(x,params[3:6], max_top=max_top) for x in range(0,max_grade)]		)
	y.append( [trianglural_func(x,params[4:7], max_top=max_top) for x in range(0,max_grade)]		)
	y.append( [trianglural_func(x,params[5:8], max_top=max_top) for x in range(0,max_grade)]		)
	y.append( [sigma_right_func(x,params[6:8], max_top=max_top) for x in range(0,max_grade)]		)

	y_vds	= [x*15 for x in range(0,(max_grade-1))]
	x_vds	= []
	x_vds.append( [member_vds[0]]*(max_grade-1))
	x_vds.append( [member_vds[1]]*(max_grade-1))
	x_vds.append( [member_vds[2]]*(max_grade-1))
	x_vds.append( [member_vds[3]]*(max_grade-1))
	x_vds.append( [member_vds[4]]*(max_grade-1))
	x_vds.append( [member_vds[5]]*(max_grade-1))
	x_vds.append( [member_vds[6]]*(max_grade-1))
	x_vds.append( [member_vds[7]]*(max_grade-1))


	x 		= range(0,max_grade)	
	
	return plt.plot(
		x, y[0], x, y[1], x, y[2], x, y[3], x, y[4], x, y[5], x, y[6], x, y[7],
		x_vds[0], y_vds, x_vds[1], y_vds, x_vds[2], y_vds, x_vds[3], y_vds, x_vds[4], y_vds, x_vds[5], y_vds, x_vds[6], y_vds, x_vds[7], y_vds
	)


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
	max_grade = 256

	
	defuzz_list = []
	for z in range(0, max_grade):
		ums 			= []
		ums.append(sigma_left_func(z,  params[0:2]))
		ums.append(trianglural_func(z, params[0:3]))
		ums.append(trianglural_func(z, params[1:4]))
		ums.append(trianglural_func(z, params[2:5]))
		ums.append(trianglural_func(z, params[3:6]))
		ums.append(trianglural_func(z, params[4:7]))
		ums.append(trianglural_func(z, params[5:8]))
		ums.append(sigma_right_func(z, params[6:8]))

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
	#plt.plot(graph[0], graph[1])

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

def show_plot_graph(graph, member_params, member_vds, image_filename_path=""):
	max = 255



	plt.bar(list(graph[0]), list(graph[1]), color=(0.6, 0.4, 0.6, 0.3))
	#plt.plot(graph[0], graph[1])

	plt_member_functions(member_params, member_vds, max_top=35000)
	
	
	image_filename 		= image_filename_path.split("\\")[-1]
	image_path			=  "\\".join(image_filename_path.split("\\")[:-1])

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



	

	src_hyst 			= build_hist(img)	
	
	#fuzzzyyyyyyyyy

	maxs				= maximums(src_hyst,6)
	print("Local max list: ",maxs)
	# photo 9737
	# member_params		= [0,10,25,45,75,140,252,255]
	# member_params		= [0,5,8,15,56,66,76,255]
	member_params		= list(maxs.keys())
	member_params.insert(0, 0)
	print("memeber params", member_params)
	member_params.append(255)
	#	this number's list is working properly on 9752 image
	#	member_params		= [0,5,8,15,56,66,76,255]
	#	member_vds			= [0,35,70,105,140,175,210,255]
	member_vds			= [0,35,70,105,140,175,210,255]

	fuzzy_image 		= get_fuzzy_image(img, member_params, member_vds)
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

	
	
	#show_graph(graph_sec_der)

	cv2.imwrite(wrt_path+image_filename, fuzzy_image)
	#print(find_equal_regions(src_hyst))
	#show_cdf_derivative(src_hyst)
	plt.plot(member_params, [0,0,0,0,0,0,0,0], 'ro')
	show_cdf_func(src_hyst)	
	show_transformation_func(member_params, member_vds)
	plt.bar(list(fuzz_hyst.keys()), list(fuzz_hyst.values()),color=(0.2, 0.4, 0.6, 0.3))
	plt.bar(list(local_max.keys()), list(local_max.values()),color=(0.6, 0.4, 0.6, 0.7))
	show_plot_graph(graph_src_hyst,  member_params, member_vds)
	
	#show_graph(graph_src_hyst, member_params, member_vds)
	#show_graph(graph_fuzz_hyst, member_params, member_vds)

	
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

