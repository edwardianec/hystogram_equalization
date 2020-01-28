import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2, itertools
import gc
import glob
import statistics
import random
import collections

#-------------------------------------------------------------------------------------------------------------
#	Операции с гистограммой
BIT_DEPTH 			= 8
BIT_DEPTH_MAX_VALUE = 2**BIT_DEPTH



def build_hist(img):
	height, width = img.shape
	hist = dict(enumerate([0]*256))
	for j in range(0,height):
		for i in range(0, width):			
			pixel_val = int(img[j,i])
			#print("pix_val:",pixel_val)
			hist[pixel_val]	= hist[pixel_val]+1

	return hist	

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

def analyze_histogram(h, member_vds):
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
	return result


def get_member_params(h, region_stat, member_vds):
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
	
	
	return sorted(result)


		



def build_first_derivative(hist):
	bit_depth 	= len(list(hist.keys())) #256 к примеру
	derivative = dict(enumerate([0]*(bit_depth)))
	j = 0
	for i in range(0, bit_depth):
		hist_val = hist[i]
		if (i > 0 and i < (bit_depth-1) ):
			derivative[j] = (hist[i+1] - hist[i-1])/2		
		j = j +1

	return derivative

def get_derivative_maxs(derivative_histogram):
	h = derivative_histogram
	bit_depth 	= len(list(h.keys())) #256 к примеру
	derivative = dict(enumerate([0]*(bit_depth)))
	maxs = []
	for i in range(1, bit_depth):
		if (h[i] < 0 and h[i-1] >= 0): maxs.append(i-1)
	return maxs

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





def maximums(graph, maximums_count=0, start=0, end=255):
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
	return maximums

def region_max_value(h, start, end):
	h_values = list(h.values())[start:end]
	max_value = max(h_values)
	max_index = start + h_values.index(max_value)
	return max_index

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


def histogram_borders(h, member_vds):
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
	return (left, right)



def calculate_median(l):
	l = sorted(l)
	l_len = len(l)
	return l[(l_len)//2]

def find_equal_regions(h):
	# данная штука делит количество всех пикселей на 7.
	# и находит те области гистограммы, где содержится 1/7 часть
	# все пикселей. Цель - разрядить гистограмму таким образом,
	# посредством смещения в ту или иную сторону.

	sum_pix = cumul_hist_val(h, BIT_DEPTH_MAX_VALUE-1 )
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
	
	print("region grade: ", region_grade)

	return region_grade


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
	y.append( [trianglural_func(x,params[6:9], max_top=max_top) for x in range(0,max_grade)]		)
	y.append( [sigma_right_func(x,params[7:10], max_top=max_top) for x in range(0,max_grade)]		)

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
	x_vds.append( [member_vds[8]]*(max_grade-1))


	x 		= range(0,max_grade)	
	
	return plt.plot(
		#x, y[0], x, y[1], x, y[2], x, y[3], x, y[4], x, y[5], x, y[6], x, y[7],
		x_vds[0], y_vds, x_vds[1], y_vds, x_vds[2], y_vds, x_vds[3], y_vds, x_vds[4], y_vds, x_vds[5], y_vds, x_vds[6], y_vds, x_vds[7], y_vds, x_vds[8], y_vds
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
		ums.append(trianglural_func(z, params[6:9]))
		ums.append(sigma_right_func(z, params[7:10]))

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

def show_first_derivative(h):
	plt.bar(list(h.keys()), list(h.values()))
	#plt.show()	

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

def show_plot_graph(graph, member_params, member_vds,  equal_regions, image_filename_path=""):
	max = 255
	#equal_regions = [0] + equal_regions + [255]
	colors = [
		(1, 0, 0, 0.4), (0, 0, 1, 0.4), (0, 1, 0, 0.4) , (0, 1, 1, 0.4),
		(1, 1, 0, 0.4), (1, 0, 1, 0.4), (0, 1, 0.5, 0.4),  (1, 0,0, 0.4)
	]

	# каждый кусок 1/6 части гистограммы нам нужно расскрасить
	for region in equal_regions:
		position = equal_regions.index(region)
		start = region[0]
		end = region[1]
		x = list(graph[0])[start:end]
		y = list(graph[1])[start:end]
		plt.bar(x, y, color=colors[position])

	#plt.bar(list(graph[0]), list(graph[1]),color=colors[7] )

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

def apply_lut(img,):
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
	return colored_image


def region_devider(h, borders, region_total, needed_dots, right):
	start = borders[0]
	h_values = h.values()
	end = borders[1]
	if (needed_dots == 1): divider = 2
	else: divider = needed_dots
	one_part = region_total/divider
	temp_sum = 0
	dots = []
	for index, count in list(enumerate(h_values))[start:end]:
		temp_sum = temp_sum + count
		if (temp_sum >= one_part): 
			temp_sum = 0
			dots.append(index)
			if (needed_dots == 1): 
				break
			else: 
				if (len(dots) == (needed_dots-1)):
					if (right): dots.append(end)
					else: dots.append(start)
					break
	return dots

def calculate_points(h, sum_pix, region_pixels, regions, h_borders):
	sorted_regions = sorted(region_pixels, reverse=True)
	total_dots = []
	for region in sorted_regions:
		this_index = region_pixels.index(region)
		percentage = (region/sum_pix)*100
		if (this_index < 4): right = True
		else: right = False
		if (percentage >= 80): dots_count = 5			
		elif (percentage >= 40): dots_count = 3
		elif (percentage >= 30): dots_count = 2
		else: dots_count = 1
		print("this_index: \t", this_index)
		print("percentage: \t", percentage)
		print("dots_count: \t", dots_count)
		print("right: \t", right)

		

		dots = region_devider(h, regions[this_index], region, dots_count, right)
		print("dots: \t", dots)
		total_dots = total_dots + dots
		if (len(total_dots) == 7): 
			break
	total_dots = [h_borders[0]] + sorted(total_dots) + [h_borders[1]]
	print("total_dots \t", total_dots)
	return total_dots




def image_statisctic(h, member_vds):
	# делим картинку на 8 частей
	# в каждой из частей определяем - какое количество
	# пикселей там присутсвует. Если в левой части гистограммы большое скопление - 
	# картинка темная, иначе - светлая.
	# Если в каждой части примерно равное количество - картинка выровненная.
	# Если большая часть в середине - то картинка не контрастная - необходимо немного
	# переместить пиксели влево и вправо, чтобы добиться небольшого контраста. 
	print("********************** image_statisctic **********************")

	sum_pix = cumul_hist_val(h, BIT_DEPTH_MAX_VALUE-1 )
	region 	= BIT_DEPTH_MAX_VALUE//8	
	regions = [[j*region, region*j+region] for j in range(0,8)]
	pixels 	= []
	h_values = list(h.values())
	for region in regions:
		start = region[0]
		end = region[1]
		pix_region = sum(h_values[start:end])
		pixels.append(pix_region)
	print("pixels in regions: {0} \t".format(pixels))
	
	borders = histogram_borders(h, member_vds)	
	total_dots = calculate_points(h, sum_pix, pixels, regions, borders)
	



	left_part 		= pixels[0]+pixels[1]+pixels[2]+pixels[3]
	dark_part 		= pixels[0]+pixels[1]
	darkest_part	= pixels[0]

	right_part 		= pixels[4]+pixels[5]+pixels[6]+pixels[7]
	light_part 		= pixels[6]+pixels[7]
	lightest_part	= pixels[7]

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
		
		
	
		
	
	print("quality: {0} \t".format(answer))
	return total_dots

def fuzzy_process(path, wrt_path):
	image_filename 		= path.split("\\")[-1]
	image_path			=  "\\".join(path.split("\\")[:-1])

	img 				= cv2.imread(path, 0)
	height, width 		= img.shape	

	src_hyst 			= build_hist(img)
	median_histogram	= get_median_histogram(src_hyst)
	median_first_derivative 	= build_first_derivative(median_histogram)
	derivative_maxs 	= get_derivative_maxs(median_first_derivative)
	target_histogram	= median_histogram
	
	equal_regions 		= find_equal_regions(src_hyst)
	

	member_regions		= [[0, 15, 32],[45,53,64],[75],[115],[],[],[],[255]]
	member_params = []
	for g in range(len(member_regions)):
		member_params		= member_params + member_regions[g] 	
	
	#member_vds			= [[0,32],[64, 96],[96, 128],[128, 160],[160, 192],[192, 224],[192, 224],[224, 256]]
	member_vds			= [0,32,64,96,128,160,192,224, 255]
	member_params	    = image_statisctic(src_hyst, member_vds)
	region_stat 		= analyze_histogram(src_hyst, member_vds)

	#member_params		= get_member_params(src_hyst,region_stat, member_vds)

	#member_params		= [6, 24, 50, 82, 105, 135,175, 212, 255]
	
	#member_params 		= [borders[0]] + [5,7, 13, 22,36,80]+ [borders[1]]
	
	print("memeber params:  \t", member_params)
	print("region_stat: \t", region_stat)
	

	fuzzy_image 		= get_fuzzy_image(img, member_params, member_vds)
	fuzz_hyst 			= build_hist(fuzzy_image)

	graph_target_hist	= [target_histogram.keys(), target_histogram.values()]

	#local_max			= find_local_max(src_hyst)
	cv2.imwrite(wrt_path+image_filename, fuzzy_image)
	""" 	colored_image = apply_lut(img)
		cv2.imwrite(wrt_path+"lut_"+image_filename, colored_image) """

	#статистика
	

	#графики
	#plt.bar(list(src_hyst.keys()), list(src_hyst.values()), color=(0, 0, 1, 0.6))
	#plt.bar(list(fuzz_hyst.keys()), list(fuzz_hyst.values()), color=(1, 0, 0, 0.6))
	#plt.show()	
	#plt.plot(member_params, [0,0,0,0,0,0,0,0,0], 'ro')

	#show_cdf_func(src_hyst)	
	#show_transformation_func(member_params, member_vds)
	#show_first_derivative(median_first_derivative)
	
	#plt.bar(list(fuzz_hyst.keys()), list(fuzz_hyst.values()),color=(0.2, 0.4, 0.6, 0.3))
	#show_plot_graph(graph_target_hist,  member_params, member_vds, equal_regions)
	
	#


#------------------------------MAIN-----------------------------------------

src_path = "D:\\resilio\\ip_lib_ed\\src_images\\dslr\\tif\\fuzzy\\src\\*.*"
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

