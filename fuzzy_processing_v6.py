import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2, itertools
import gc
import glob
import statistics
import random
import collections
from pathlib import Path


import fuzzyfication

#-------------------------------------------------------------------------------------------------------------
#	Операции с гистограммой
BIT_DEPTH 			= 8
BIT_DEPTH_MAX_VALUE = 2**BIT_DEPTH



""" def build_hist(img):
	height, width, colors = img.shape
	hist = dict(enumerate([0]*256))
	for j in range(0,height):
		for i in range(0, width):			
			pixel_val = int(img[j,i])
			#print("pix_val:",pixel_val)
			hist[pixel_val]	= hist[pixel_val]+1

	return hist	 """

def get_histogram(img):
	hist = cv2.calcHist([img],[0], None, [256], [0,256])
	hist = hist.astype(int)
	return hist	

def find_histogram_borders(h):
	for index, el in enumerate(h):
		if (el>0):
			left = index
			break
	for index, el in enumerate(h):
		if (el>40):
			right = index

	return (left, right)	

def gaus_filter(img):
	""" 	kernel_gausian 		= [
		[1/16,	2/16,	 1/16],
		[2/16, 4/16,	 2/16],
		[1/16,	2/16,	 1/16]
	] """
	kernel_gausian 		= [
		[1/273,		4/273,	 7/273,		4/273, 	1/273],
		[4/273, 	16/273,	 26/273,	16/273, 4/273],
		[7/273,		26/273,	 41/273,	26/273, 7/273],
		[4/273,		16/273,	 26/273,	16/273, 4/273],
		[1/273,		4/273,	 7/273,		4/273, 	1/273]
	]
	kernel_gausian_mat 	= np.array(kernel_gausian, dtype=float);
	gausian_image 		= cv2.filter2D(img,-1,kernel_gausian_mat)
	return gausian_image


def get_median_histogram(h):
	bit_depth 	= len(list(h.keys())) #256 к примеру
	filtered 	= dict(enumerate([0]*(bit_depth)))
	for i in range(0, bit_depth):	
		if (i < 3):
			if (i==0):	filtered[i] = h[0]
			elif (i==1): filtered[i] = h[1]
			elif (i==2): filtered[i] = statistics.median([ h[i-2], h[i-1], h[i] ])
		else: filtered[i] = statistics.median([ h[i-3], h[i-2], h[i-1], h[i] ])
	return filtered	

def normalize_image(img, h):
	h_min_indx, h_max_indx	= find_histogram_borders(h)	

	height, width 	= img.shape	
	image_res = np.zeros(shape=[height, width, 1], dtype=np.uint8)
	for i in range(0,width):
		for j in range(0, height):
			image_res[j,i] 	= ((img[j,i] - h_min_indx)/(h_max_indx-h_min_indx))*(len(h)-1)	
	#cv2.imshow("normilized",image_res)
	return image_res

def trashold_calculation(img):
	h1			= get_histogram(img)	
	h2		 	= np.sort(h1, axis=0, order=None)[::-1] # по убыванию
	tau1 		= h2[0]
	tau2 		= h2[255]
	i 			= 1
	while (tau1 > tau2):
		tau = tau1
		tau1 = (tau1+h1[i])/2
		tau2 = (tau2+h2[255-i])/2
		i = i+1
	return tau

def merge_insignificant_bins(h, img):
	tau = trashold_calculation(img)
	print("TAU IS :",tau)
	h2 = np.zeros(shape=[256,1], dtype=np.uint32)
	for i in range(0, len(h)-1):
		if (h[i]< tau):
			h2[i] = 0
			h2[i+1] = h[i] + h[i+1]
		else: h2[i] = h[i]
	return h2

""" def cumul_hist_val(histogram, grey):
	#print(histogram)
	cum = 0
	for i in range(0, grey):
		
		cum = cum + histogram[i]
	return cum """

""" def cumulative_function(histogram):
	cum_dict = {}
	for i in range(0,256):
		cum_dict[i] =  cumul_hist_val(histogram,i)
	
	return (cum_dict.keys(), cum_dict.values()) """

""" def analyze_histogram(h, member_vds):
	total_pixels = cumul_hist_val(h, BIT_DEPTH_MAX_VALUE-1 )
	# я не считаю крайние правила слева и справа по одному
	rights = len(member_vds)-2
	pixels_per_right = total_pixels//rights


	h_values = list(h.values())
	pixels_per_region = {}
	for i in range(0, len(member_vds)-1):
		start = member_vds[i]
		end = member_vds[i+1]
		region_pixels = 0
		for el in h_values[start:end]:
			region_pixels = region_pixels + el
		#pixels_per_region.append({i:region_pixels, round(region_pixels/pixels_per_right, 2)]})
		pixels_per_region[(i, start,end, round(region_pixels/pixels_per_right, 2))] = region_pixels
	pixels_per_region = {k: v for k, v in sorted(pixels_per_region.items(), key=lambda item: item[1], reverse=True)}
	print("pixels_per_region: \t",pixels_per_region)

	dots = 0
	result = {}
	for key in pixels_per_region:
		points = pixels_per_region[key]//pixels_per_right
		#points = round(pixels_per_region[key]/pixels_per_right)
		
		if (points == 0 ): points = 1
		
		if (dots+points <= 6):
			dots = dots + points
			result[key] = [pixels_per_region[key], points]
	return result """

""" def get_member_params(h, region_stat, member_vds):
	h_values = list(h.values())
	borders = histogram_borders(h, member_vds)
	result = [borders[0]]
	for key in region_stat:
		keys = list(region_stat.keys())
		start = key[1]
		end = key[2]
		region_points = region_stat[key][1]
		if (key==keys[0] and start==0): start = borders[0]
		if (key==keys[-1] and end==255): end = borders[1]
		#result = result + random.choices(range(start,end), k=region_stat[key][1])
		# maxs = list(maximums(h, maximums_count=region_stat[key][1], start=start, end=end).values())
		maxs = []
		dist = (end-start)//(region_points+1)
		for i in range(1, region_points+1 ):
			maxs.append(start+dist*i)
		print("maxs: \t",maxs)
		result = result + maxs
	result.append(borders[1])
	
	
	return sorted(result) """

""" def build_first_derivative(hist):
	bit_depth 	= len(list(hist.keys())) #256 к примеру
	derivative = dict(enumerate([0]*(bit_depth)))
	j = 0
	for i in range(0, bit_depth):
		hist_val = hist[i]
		if (i > 0 and i < (bit_depth-1) ):
			derivative[j] = (hist[i+1] - hist[i-1])/2		
		j = j +1

	return derivative """

""" def get_derivative_maxs(derivative_histogram):
	h = derivative_histogram
	bit_depth 	= len(list(h.keys())) #256 к примеру
	derivative = dict(enumerate([0]*(bit_depth)))
	maxs = []
	for i in range(1, bit_depth):
		if (h[i] < 0 and h[i-1] >= 0): maxs.append(i-1)
	return maxs """

""" def maximums(graph, maximums_count=0, start=0, end=255):
	# данный метод позволяет найти максимумы на графике
	# и выбрать из него только необходимое количество из
	# всех максимумов. При этом максимумы выбираются начиная
	# с самых больших значений. 
	maximums = {}
	previous = 0
	#print("start, end: ", start, end)
	for i in range(start+1, end):
		current = graph[i] - graph[i-1]
		#print("graph[",i,"]:", graph[i])
		#print(current)
		if (current < 0 and previous > 0): maximums[i-1] = graph[i-1]
		previous = current

	# После того, как мы нашли максимумы, мы должны упорядичить эти
	# максимумы в порядке убывания, после чего выбрать только необходимое нам количество,
	# определенное в переменной maximums_count
	maximums	= {k: v for k, v in sorted(maximums.items(), key=lambda item: item[1], reverse=True)}
	if (maximums_count):
		maximums 	= dict(itertools.islice(maximums.items(),maximums_count))
	maximums	= dict(sorted(maximums.items()))
	return maximums """

""" def region_max_value(h, start, end):
	h_values = list(h.values())[start:end]
	max_value = max(h_values)
	max_index = start + h_values.index(max_value)
	return max_index """

""" def find_local_max(h):
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
	return local_max """

""" def histogram_borders(h, member_vds):
	keys = list(h.keys())
	for el in h:
		if (h[el]>20):
			left = el
			break
	for el in reversed(keys):
		if (h[el]>20):
			right = el
			break	
	if (right < member_vds[1]): right = member_vds[1]	
	if (left > member_vds[-2]): left = member_vds[-2]	
	return (left, right) """

""" def calculate_median(l):
	l = sorted(l)
	l_len = len(l)
	return l[(l_len)//2] """


#-------------------------------------------------------------------------------------------------------------

# 	графики
#  

""" def show_cdf_func(h):
	graph = cumulative_function(h)
	
	plt.plot(list(graph[0]),list(graph[1]) )
	plt.show()	 """

""" def show_cdf_derivative(h):
	cdf_graph 	= cumulative_function(h)
	cdf_dict	= dict(zip(cdf_graph[0], cdf_graph[1]))
	deriv_graph = build_first_derivative(cdf_dict)
	plt.plot(list(deriv_graph.keys()),list(deriv_graph.values()) )
	plt.show()	 """	

""" def show_transformation_func(member_params, member_vds ):
	plt.plot(list(range(0,256)), get_defuzzification_list(member_params, member_vds), list(range(0,256)), list(range(0,256)))
	plt.show() """

""" def show_first_derivative(h):
	plt.bar(list(h.keys()), list(h.values()))
	#plt.show()	 """

""" def quality(region_percent):
	left_part 		= region_percent[0]+region_percent[1]+region_percent[2]+region_percent[3]
	dark_part 		= region_percent[0]+region_percent[1]
	darkest_part	= region_percent[0]

	right_part 		= region_percent[4]+region_percent[5]+region_percent[6]+region_percent[7]
	light_part 		= region_percent[6]+region_percent[7]
	lightest_part	= region_percent[7]

	print("regions: \t", regions)
	print("left_part: \t",left_part)
	print("dark_part: \t",dark_part)
	print("darkest_part: \t",darkest_part)
	print("----------------- \t")
	print("right_part:\t", right_part)
	print("light_part:\t", light_part)
	print("lightest_part:\t", lightest_part)


	if (left_part > 70 ):		
		answer = "картинка темная"
		if (dark_part > left_part//2):
			answer = "картинка темноватая"
			if (darkest_part > dark_part//2): answer = "картинка оч темная"			
	elif (right_part > 70 ):		
		answer = "картинка светлая"
		if (light_part > right_part//2):
			answer = "картинка светловатая"
			if (lightest_part > light_part//2): answer = "картинка оч светлая"
	else:	answer = "картинка нормас"	
	
	print("quality: {0} \t".format(answer)) """

#-----------------------------------------

""" def apply_lut(img,):
	height, width 	= img.shape	
	colored_image = np.zeros(shape=[height, width, 3], dtype=np.uint8)
	for i in range(0,width):
		for j in range(0, height):
			pixel_val = img[j,i]	
			if (pixel_val < 200):						
				new_val = 	[0,0, pixel_val]
			else:
				new_val = 	[pixel_val,pixel_val, pixel_val]	
			colored_image[j,i] = new_val	
	return colored_image """

def region_devider(h, borders, region_total, needed_dots, right, this_index):
	start = borders[0]
	end = borders[1]
	if (needed_dots == 1): divider = 2
	else: divider = needed_dots
	print("region_total :",region_total)
	one_part = region_total/divider
	print("one_part :",one_part)

	temp_sum = 0
	dots = []

	for index, count in list(enumerate(h))[start:end]:
		temp_sum = temp_sum + count
		if (temp_sum >= one_part): 
			if (temp_sum > one_part*1.5):
				temp_sum = temp_sum - one_part
			else: temp_sum = 0

			dots.append(index)
			if (needed_dots == 1):
				break
			print("len(dots) :",len(dots))
			print("needed_dots-1 :",needed_dots-1)
			if (len(dots) == (needed_dots-1)):
				if (right): dots.append(end)
				else: dots.append(start)
				break
	return dots

def calculate_points(h, sum_pix, region_pixels, regions, member_vds):
	borders = find_histogram_borders(h)

	left_border = member_vds[1] if borders[0] > member_vds[1] else borders[0]
	right_border = member_vds[-2] if borders[1] < member_vds[-2]   else borders[1]
	sorted_regions = sorted(region_pixels, reverse=True)
	total_dots = []
	for region in sorted_regions:
		this_index = region_pixels.index(region)
		percentage = int((region/sum_pix)*100)
		if (this_index < 4): right = True
		else: right = False
		dots_count = 1
		if (percentage >= 60): dots_count = 4
		elif (percentage >= 50): dots_count = 3
		elif (percentage >= 40): dots_count = 2
		elif (percentage >= 30): dots_count = 1		
		
		
		print("--------------")
		print("number: 	", this_index)
		print("percent: 	", percentage, " %")
		print("dots_count:	", dots_count)
		print("right:		", right)

		

		dots = region_devider(h, regions[this_index], region, dots_count, right, this_index)
		print("dots:		", dots)
		total_dots = total_dots + dots
		if (len(total_dots) == 7): 
			break
	print("borders	", borders)
	total_dots = [left_border] + sorted(total_dots) + [right_border]
	print("total_dots	", total_dots)
	return total_dots


def image_statisctic(h, member_vds, sum_pix):
	# делим картинку на 8 частей
	# в каждой из частей определяем - какое количество
	# пикселей там присутсвует. Если в левой части гистограммы большое скопление - 
	# картинка темная, иначе - светлая.
	# Если в каждой части примерно равное количество - картинка выровненная.
	# Если большая часть в середине - то картинка не контрастная - необходимо немного
	# переместить пиксели влево и вправо, чтобы добиться небольшого контраста. 
	print("********************** image_statisctic **********************")

	region 	= BIT_DEPTH_MAX_VALUE//8	
	regions = [[j*region, region*j+region] for j in range(0,8)]
	pixels 	= []
	for region in regions:
		start = region[0]
		end = region[1]
		pix_region = sum(h[start:end])
		pixels.append(pix_region)
	print("pixels in regions: {0} \t".format(pixels))
	
	#borders = histogram_borders(h, member_vds)	
	total_dots = calculate_points(h, sum_pix, pixels, regions, member_vds)

	return total_dots

def find_equal_regions(h, sum_pix):
	# данная штука делит количество всех пикселей на 7.
	# и находит те области гистограммы, где содержится 1/7 часть
	# все пикселей. Цель - разрядить гистограмму таким образом,
	# посредством смещения в ту или иную сторону.

	region = sum_pix//8
	print("pixel_counter: {0} pixels in image; 1/6 part of pixels: {1}".format(sum_pix, region))
	cum = 0
	j = 0
	region_grade = []
	previos_val = 0
	for i in range(0, BIT_DEPTH_MAX_VALUE):
		cum = cum + h[i]
		#print("cum=",cum)
		if (cum >= region): 
			cum = 0
			region_grade.append([previos_val, i-1])
			previos_val = i-1			
			if (j == 5):
				region_grade.append([previos_val, 255])
			j = j+1
	
	print("equal regions: ", region_grade)

	return region_grade

def show_plot_graph(h, member_params, member_vds,  equal_regions, image_filename="", wrt_path=""):
	max = 255
	#equal_regions = [0] + equal_regions + [255]
	colors = [
		(1, 0, 0, 0.4), (0, 0, 1, 0.4), (0, 1, 0, 0.4) , (0, 1, 1, 0.4),
		(1, 1, 0, 0.4), (1, 0, 1, 0.4), (0, 1, 0.5, 0.4),  (1, 0,0, 0.4)
	]

	keys = list(range(0,len(h)))
	values = list(h.ravel())
	# каждый кусок 1/8 части гистограммы нам нужно расскрасить
	for region in equal_regions:
		position = equal_regions.index(region)
		start = region[0]
		end = region[1]
		x = keys[start:end]
		y = values[start:end]
		plt.bar(x, y, color=colors[position])

	#plt.bar(list(graph[0]), list(graph[1]),color=colors[7] )

	fuzzyfication.plt_member_functions(member_params, member_vds, max_top=35000)

	if (wrt_path):
		fig_wrt_path = wrt_path+"\\fig\\"
		Path(fig_wrt_path).mkdir(parents=True, exist_ok=True)
		plt.savefig(fig_wrt_path+image_filename+"_figure.png", dpi=500)
	else:		
		plt.show()
	plt.close()


def show_graphs(h1, h2, member_params, member_vds, total_pixels, image_filename="", wrt_path=""):
	#статистика
	
	equal_regions = find_equal_regions(h1, total_pixels)
	#графики
	#plt.bar(list(h1.keys()), list(h1.values()), color=(0, 0, 1, 0.6))
	#plt.bar(list(h2.keys()), list(h2.values()), color=(1, 0, 0, 0.6))
	#plt.show()	
	plt.plot(member_params, [0,0,0,0,0,0,0,0,0], 'r^')

	#show_cdf_func(src_hyst)	
	#show_transformation_func(member_params, member_vds)
	#show_first_derivative(median_first_derivative)
	plt.bar(list(range(0,len(h2))),list(h2.ravel()), color=(0.2, 0.4, 0.6, 0.3))
	
	#show_plot_graph([h1.keys(), h1.values()],  member_params, member_vds, equal_regions,image_filename, wrt_path )
	show_plot_graph(h1,  member_params, member_vds, equal_regions,image_filename, wrt_path )

def show_histogram(h):
	#plt.hist(img.ravel(),256,[0,256])
	plt.bar(list(range(0,len(h))),list(h.ravel()))
	plt.show()


def fuzzy_main_process(path):
	image_filename 			= path.split("\\")[-1]
	image_path				=  "\\".join(path.split("\\")[:-1])
	wrt_path				= image_path+"\\processed\\"
	Path(wrt_path).mkdir(parents=True, exist_ok=True)

	img 					= cv2.imread(path, 0)
	height, width 			= img.shape	




	np_src_histogram		= get_histogram(img)
	#show_histogram(np_src_histogram)

	#gaus_img				= gaus_filter(img)
	#gaus_img_h				= get_histogram(gaus_img)
	#show_histogram(gaus_img_h)

	#normalized_image		= normalize_image(img, np_src_histogram)
	#normalized_histogram 	= 	get_histogram(normalized_image)
	#merged_histogram 		= 
	src_hyst 				= np_src_histogram # merge_insignificant_bins(normalized_histogram, normalized_image)
	#show_histogram(src_hyst)
	#median_histogram	= get_median_histogram(src_hyst)
	#median_first_derivative 	= build_first_derivative(median_histogram)
	#derivative_maxs 	= get_derivative_maxs(median_first_derivative)
	#target_histogram	= median_histogram
	print("			")
	print("			")
	print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
	print("			")
	print("filename: \t", image_filename)
	print("			")
	print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
	
	member_vds				= [0,32,64,96,128,160,192,224, 255]
	member_params	    	= image_statisctic(src_hyst, member_vds, height*width)
	#member_params	    	=   [0, 12, 20, 26, 32, 46, 86, 120, 224]
	fuzzy_image 			= fuzzyfication.get_fuzzy_image(img, member_params, member_vds)
	fuzz_hyst 				= get_histogram(fuzzy_image)

	clahe 				= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image 		= clahe.apply(fuzzy_image)


	#local_max				= find_local_max(src_hyst)
	cv2.imwrite(wrt_path+image_filename, clahe_image)
	""" 	colored_image = apply_lut(img)
		cv2.imwrite(wrt_path+"lut_"+image_filename, colored_image) """
	#show_graphs(src_hyst, fuzz_hyst, member_params, member_vds, height*width,image_filename, wrt_path )

def clahe_method(path):
	image_filename 			= path.split("\\")[-1]
	image_path				=  "\\".join(path.split("\\")[:-1])
	wrt_path				= image_path+"\\processed\\clahe\\"
	Path(wrt_path).mkdir(parents=True, exist_ok=True)

	img 					= cv2.imread(path, 0)
	height, width 			= img.shape	

	clahe 				= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
	clahe_image 		= clahe.apply(img)
	cv2.imwrite(wrt_path+"clahe_"+image_filename, clahe_image)




#------------------------------MAIN-----------------------------------------

#src_path = "D:\\image_library\\fuzzy_logic\\test\\"
src_path =  "D:\\image_library\\fuzzy_logic\\room\\video\\"
#src_path =  "D:\\image_library\\fuzzy_logic\\street\\yes\\"
#src_path =  "D:\\image_library\\fuzzy_logic\\kodak\\"


images 		= glob.glob(src_path+"*.*")
img_amount 	= len(images)
img_counter = 0

print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("PROCESSING...")
print("TOTAL_IMAGES: ", img_amount)


for image_path in images:
	img_counter = img_counter + 1	
	print("				")
	print("-	-	-	-	-	-	-")
	print("IMAGE {0} of {1}".format(img_counter, img_amount))
	fuzzy_main_process(image_path)
	#clahe_method(image_path)

#concat_images = np.concatenate( (my_resize(img4,50), my_resize(img4, 50)), axis=1)
#cv2.imshow('modified_image',concat_images )
#my_subplots(img4, img4)

gc.collect()
cv2.waitKey(0)
cv2.destroyAllWindows()

