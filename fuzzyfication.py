
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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


def get_fuzzy_image(img, params, vds):
	height, width, = img.shape[0:2]	
	fuzzy_image 	= np.zeros(shape=[height, width], dtype=np.uint8)
	fuzzy_list		= get_defuzzification_list(params, vds)
	for i in range(0,width):
		for j in range(0, height):
			pixel_val = int(img[j,i]			)
			new_val = 	fuzzy_list[pixel_val]	
			fuzzy_image[j,i] = new_val
	return fuzzy_image