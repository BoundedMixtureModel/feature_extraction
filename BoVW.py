# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:55:46 2019

@author: Ravi
"""

import cv2
from sklearn.cluster import KMeans
import glob
import numpy as np
import os


sift_data =[]
lbp_data = []

def read_files(file, count):
    data = []
    print('count', count)
    #add your path for loading images
    # load each frame of an action
    files = glob.glob ("add path to extracted images /*.jpg")
    file = (files[i] for i in range(count))  
    for myFile in file:
        print(myFile)
        image = cv2.imread (myFile)
        data.append (image)
    return data

# extracting the features
def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

#building the histogram
def build_histogram(descriptor_list, cluster_alg):
    #print('centers', cluster_alg.cluster_centers_.shape)
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    #print('histogram', histogram.shape)
    #print("list",descriptor_list.shape)
    cluster_result =  cluster_alg.predict(descriptor_list)
    #print('cluster result',cluster_result.shape)
    for i in cluster_result:
        histogram[i] += 1.0
    #print(histogram)
    return histogram

def build_sift_BoVW(data, extractor, clusters):
    #initializing the K_means
    print('in sift')
    kmeans = KMeans(n_clusters=clusters)
    processed_image = []
    desc_list = []
    for image in data:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp,desc = features(image, extractor)
        desc_list.append(desc)
    print(len(desc_list))
    vStackk = np.array(desc_list[0])
    for remaining in desc_list:
    	vStackk = np.vstack((vStackk, remaining))
    kmeans.fit(vStackk)
    #print(len(data))
    for des in desc_list:
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #kp,desc = features(image, extractor)
        print(desc.shape)        
        if(des is not None and des.shape[0] >= clusters ):
            #kmeans.fit(desc)
            histogram = build_histogram(des, kmeans)
            processed_image.append(histogram)
        else:
            ignored = []
            ignored.append(desc)
            print('ignored', len(ignored))

    vStack = np.array(processed_image[0])
    for remaining in processed_image:
    	vStack = np.vstack((vStack, remaining))
    sift_data.append(np.array(processed_image))    
    return sift_data

def build_lbp_BoVW(data):
    
    lbp_descriptors = []
    lbp = LBP(56,3)
    for image in data:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        result = lbp.describe(image)
        eps=1e-7
        result = result.astype("float")
        result /= (result.sum() + eps)
        lbp_descriptors.append(result)
   
    lbp_data.append(np.array(lbp_descriptors))         
    return lbp_data

def init(option, count, clusters):
    sift_data.clear()
    lbp_data.clear()    
    file_names = os.listdir("folder path to each activity img folder")
    extractor = cv2.xfeatures2d.SIFT_create(nfeatures=150)
    for file in file_names:
        print("input for read file",file)
        data = read_files(file, count)
        if option:
            print("extraction using sift")
            BoVW = build_sift_BoVW(data, extractor, clusters)
        else:
            print("extraction using lbp")
            BoVW = build_lbp_BoVW(data)
        print('before extract return')
    return   BoVW

#bag = init(0, 100 ,30)
#bag1 = np.sum(bag[0], axis = 1)