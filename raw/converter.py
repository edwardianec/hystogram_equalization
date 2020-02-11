from bitstring import BitArray
import tifffile as tiff
import numpy as np
import cv2

def get_binary(path):
	n = 8
	with open(path, 'rb') as byte_reader:
		file = byte_reader.read()
	
	binary_file = BitArray(bytes=file).bin
	
	#byte_list = [''.join(reversed(binary_file[i:i+n])) for i in range(0, len(binary_file), n)]
	byte_list = [binary_file[i:i+n] for i in range(0, len(binary_file), n)]


	k = 14

	byte_list = [i[::-1] for i in byte_list]
	byte_srt = ''.join(byte_list)
	
	byte_list_14 = [''.join(reversed(byte_srt[i:i+k])) for i in range(0, len(byte_srt), k)]

	return byte_list_14

def create_tiff(path, raw, width, height):
	new_image 	= np.zeros(shape=[height, width], dtype=np.uint16)
	num = 0
	for i in range(0,height):
		for j in range(0, width):	
			new_image[i,j] = int(raw[num],2)	
			num = num + 1
	tiff.imsave(path, new_image)

def create_test_tiff(width, height):
	new_image 	= np.zeros(shape=[height, width], dtype=np.uint16)
	num = 0
	for i in range(0,height):
		for j in range(0, width):	
			print(num)
			new_image[i,j] = num
			num = num + 1000
	return new_image

def tiff_to_raw(wrt_path, tiff_path):
	img = cv2.imread(tiff_path,-1)
	print(img[0][0])
	#cv2.imshow('16bit TIFF', img)
	#cv2.waitKey()

def extractor(org_bin, base_bin, width, height):
	new_image 	= np.zeros(shape=[height, width], dtype=np.uint16)
	num = 0
	print(org_bin[0])
	print(org_bin[1])
	print(org_bin[2])
	for i in range(0,height):
		for j in range(0, width):
			a = int(org_bin[num],2)
			b = int(base_bin[num],2)

			
			res = 0 if a-b+8000 <0 else a-b+8000
			
			new_image[i,j] = res
			#print("a,b,res,new_image[i,j] ",a,b,res,new_image[i,j])
			#print("res",res)
			#print("new_image[{0},{1}]={2}".format(i,j,new_image[i,j]))
			#
			num = num + 1
	return new_image


	




height = 512
width = 640
org_path = "src\\2\\org.raw"
base_path = "src\\2\\base.raw"
wrt_file = "src\\2\\res.tiff"

org_bin = get_binary(org_path)
base_bin = get_binary(base_path)


extracted_image = extractor(org_bin, base_bin, width, height)
tiff.imsave(wrt_file, extracted_image)
