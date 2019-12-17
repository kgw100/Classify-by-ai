import os
import sys
import binascii
import cv2
import numpy as np 
import shutil
from numpy import argmax 
from PIL import Image

def check(byte):
    if len(byte) != 0:            
        return ord(byte)
    else:
        return 0
def Dataization(img_path): 
    image_w = 64
    image_h = 64
    img = cv2.imread(img_path) 
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0]) 
    return (img/256) 

def getBinaryData(filename):
    binaryValues = []
    file = open(filename, "rb")
    data = file.read(1)  # read byte by byte
    while data !=b"":
        try:
            binaryValues.append(ord(data))  # store value to array
        except TypeError:
            pass
        data = file.read(1)  # get next byte value
    return binaryValues

def createGreyScaleImageSpecificWith(dataSet,outputfilename):
    image = Image.new('L', (30,120))
    image.putdata(dataSet)
    imagename = outputfilename+".png"
    image.save(imagename)
    print (imagename+" Greyscale image created")

HEX_LIST = ['0x00', '0x17', '0x20', '0x2F', '0x6D', '0x92', '0xA4', '0xD4' ,'0xE8', '0x0B',
            '0x0D', '0x13', '0x2B', '0x35', '0xB0', '0xB8', '0xC2', '0xD8', '0x01', '0x02', 
            '0x03', '0xC0', '0xFE', '0xFF', '0x73', '0x75', '0x61', '0x6E', '0x6F', '0x70',
            '0x83', '0x84', '0x24', '0x2A', '0x48', '0x49', '0x4A', '0x52', '0x54', '0x55', 
            '0x8E', '0x95', '0xA5', '0xA9', '0xAA', '0x05', '0x08', '0x10', '0x15', '0x18',
            '0x1D', '0x23', '0x25', '0x2D', '0x4B', '0xA0', '0xA8', '0xAD', '0xB5', '0xC4', 
            '0xC8', '0xD0', '0xD2', '0xDB', '0xE2', '0xE7', '0xEB', '0xEF', '0x7F', '0x80',
            '0x91', '0xBF', '0x06', '0x09', '0x21', '0x30', '0x85', '0x88', '0x98', '0xC1',
            '0x31', '0x36', '0x38', '0x3A', '0x3C', '0x3E', '0x47', '0x63', '0x64', '0x65',
            '0x66', '0x69', '0x6C', '0x72', '0x74', '0x76', '0x78', '0x79', '0x81', '0x82',
            '0xBC', '0xDC', '0xE1', '0xF8', '0x14', '0x1B', '0x34', '0x40', '0x5B', '0x5C', 
            '0x5E', '0x5F', '0x7C', '0x7E', '0x8B', '0xAB', '0xAE', '0xAF', '0xB6', '0xB7']
#training set
dir_list = ['h264_fragment_trainingset', 'jpg_fragment_trainingset', 'png_fragment_trainingset', 'wav_fragment_trainingset', 'pdf_fragment_trainingset', 'hwp_fragment_trainingset']


#test set
#dir_list = ['png_fragment_testset', 'h264_fragment_testset', 'h265_fragment_testset', 'jpg_fragment_testset', 'wav_fragment_testset']

for xxe in dir_list:
    path = r"C:\Users\yoon\Desktop\dataset\{0}".format(xxe)
    save_path = r"C:\Users\yoon\Desktop\CNNtest\trainingset_output\{0}_feature".format(xxe)
    #save_path = r"C:\Users\yoon\Desktop\CNNtest\test_output\{0}_feature".format(xxe)
    total_lst = [0 for _ in range(256)]
    file_lst = []
    file_num = 100 #파일 개수 설정
    file_cnt = 0
    print("파일 탐색중..")
    filenames = os.listdir(path)
    print("디렉토리 내의 파일갯수 : {0}".format(len(filenames)))

    if len(filenames) < file_num:
        sys.exit()

    for filename in filenames:
        file_lst.append(filename)
        file_cnt+=1
        if file_cnt == file_num:
            break
    print("파일 탐색완료!")
    name_index = 0
    for x in file_lst:
                      test_filename = path+r"\%s"%(x)
                       #파일의 1바이트 씩 읽어서 갯수 체크
                      lst = [0 for _ in range(256)]

                      with open(test_filename, "rb") as f:
                                   byte = f.read(1) 
                                   while byte :
                                           lst[check(byte)] += 1  
                                           byte = f.read(1)
                                   for index in range(256):
                                        total_lst[index] += lst[index]    
                      name_index += 1      
                      fn2 = '\{0}_{1}'.format(path[30:],name_index)
                      save_file = save_path + fn2
                      with open(save_file, 'wb') as fb:
                                     for xxe2 in HEX_LIST:
                                                     hex_value = int('{0}'.format(xxe2),16)
                                                     input_value = lst[hex_value]
                                                     if input_value >= 255 :
                                                            input_value = 255
                                                     input = chr(input_value)
                                                     for yy in range(30):
                                                            fb.write(input_value.to_bytes(1,byteorder='big'))
    print("{0}피쳐 추출완료".format(xxe))


for xxe in dir_list:
    if __name__=="__main__":
        src = [] 
        name = [] 
        test = []
        print("strart")

        image_dir = r'C:\Users\yoon\Desktop\CNNtest\trainingset_output\{0}_feature'.format(xxe) #test file path
        #image_dir = r'C:\Users\yoon\Desktop\CNNtest\test_output\{0}_feature'.format(xxe) #test file path
    
        #grayscale 변화
        for i in os.listdir(image_dir):
            file_full_path=image_dir + "/" + i
            path=os.path.dirname(file_full_path)
            base_name=os.path.splitext(os.path.basename(file_full_path))[0]
            outputFilename=os.path.join(path,base_name)

            binaryData=getBinaryData(file_full_path)
            createGreyScaleImageSpecificWith(binaryData, outputFilename)

            src.append(image_dir + "/" + i) 
            name.append(i)
            test.append(Dataization(image_dir + "/" + i + ".png"))

        print("파일 생성 완료")
        print("파일 변환")




