# -*- coding: utf-8 -*-
#	train1  test1 	原版
#	train1_1	test1_1		精简版
#	train1_2	test1_2		mat版
import os

root = 'D:/Project_File/py_Project/Alexnet/'
txt1 = root+'train1.txt'
txt1_1 = root+'train1_2.txt'
txt2 = root+'test1.txt'
txt2_1 = root+'test1_2.txt'

fh1 = open(txt1, 'r')
fh2 = open(txt2, 'r')
fh1_1 = open(txt1_1,'w')
fh2_1 = open(txt2_1,'w')
imgs_data1 = []
imgs_data2 = []

for line in fh1:
	line = line.strip('\n')	#移除字符串头尾指定的字符（默认为空格）
	line = line.rstrip()	#删除 string 字符串末尾的指定字符（默认为空格）
	words = line.split()	#通过指定分隔符对字符串进行切片,默认为所有的空字符,包括空格 换行(\n) 制表符(\t)等
	imgs_data1.append([words[0],int(words[1])])
fh1.close() 

for line in fh2:
	line = line.strip('\n')	#移除字符串头尾指定的字符（默认为空格）
	line = line.rstrip()	#删除 string 字符串末尾的指定字符（默认为空格）
	words = line.split()	#通过指定分隔符对字符串进行切片,默认为所有的空字符,包括空格 换行(\n) 制表符(\t)等
	imgs_data2.append([words[0],int(words[1])])
fh2.close() 

for ip in imgs_data1:
	ip[0] = ip[0].replace('/home/hbz/PycharmProjects/test_flower/', '')
	ip[0] = ip[0].replace('jpg', 'mat')
	ip[0] = ip[0].replace('segmim', 'hog_segmim',1)		#只替换一次
for ip in imgs_data2:
	ip[0] = ip[0].replace('/home/hbz/PycharmProjects/test_flower/', '')
	ip[0] = ip[0].replace('jpg', 'mat')
	ip[0] = ip[0].replace('segmim', 'hog_segmim',1)		#只替换一次
for ip in imgs_data1:  
	fh1_1.write(ip[0])
	fh1_1.write("\t")
	fh1_1.write(str(ip[1]))
	fh1_1.write('\n')
fh1_1.close() 

for ip in imgs_data2:  
	fh2_1.write(ip[0])
	fh2_1.write("\t")
	fh2_1.write(str(ip[1])) 
	fh2_1.write('\n')
fh2_1.close() 

