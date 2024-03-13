import numpy as np
import cv2 as cv
import random
import statistics as st

# to extract the background, the largest white object in our image should be the background

def extractBackground(image):
    v = 245
    height, width = image.shape
    eqlist = []
    label = 0
    labelmatrix = np.zeros(image.shape, dtype='uint16')
    for x in range(0,height):
        for y in range(0,width):
            if image[x][y] > v:
                neighbors = []
                # taking current pixel and inserting its up and left labels in a list
                if x > 0 and image[x-1][y] > v:
                    neighbors.append(labelmatrix[x-1][y])
                if y > 0 and image[x][y-1] > v:
                    neighbors.append(labelmatrix[x][y-1])
                if (y > 0 and x > 0) and image[x-1][y-1] > v:
                    neighbors.append(labelmatrix[x-1][y-1])
                if (y < width-1 and x > 0):
                    if image[x-1][y+1] > v:
                        neighbors.append(labelmatrix[x-1][y+1])

                if not neighbors: # if no neighbors in the list
                    label+=1
                    labelmatrix[x][y] = label
                    eqlist.append(label)
                elif len(neighbors) == 1: # if only one label then copy that label on that coordinate
                    labelmatrix[x][y] = neighbors[0]
                elif len(neighbors) > 1:  # if more than one label
                    if len(set(neighbors)) == 1: # if all having same label
                        labelmatrix[x][y] = neighbors[0]# assign that label
                    else:
                        # assign label based on priority (larger label has higher priority)
                        labelmatrix[x][y] = max(neighbors)
                        labelmatrix[labelmatrix == min(neighbors)] = max(neighbors)

    eqlist = np.delete(np.unique(labelmatrix), np.where(np.unique(labelmatrix) == 0))
    arr_flat = labelmatrix.flatten()

    # Get unique values and their counts
    unique_values, counts = np.unique(arr_flat, return_counts=True)

    # Sort unique values by counts in descending order
    sorted_indices = np.argsort(-counts)
    unique_values_sorted = unique_values[sorted_indices]

    # Find the index of the maximum count
    max_count_index = np.argmax(counts)

    # The most frequent value
    most_frequent_value = unique_values[max_count_index]

    # Second most frequent value
    second_most_frequent_value = None

    # Loop through sorted unique values to find the second most frequent value
    for val in unique_values_sorted:
        if val != unique_values[sorted_indices[0]]:
            second_most_frequent_value = val
            break

    backgroundcolor = second_most_frequent_value

    # now remove the background and color it 0,0,0
    # we have the image in 'image' variable
    for x in range(0,height):
        for y in range(0,width):
            if (labelmatrix[x][y] == backgroundcolor) & (image[x][y] > v):
                image[x][y] = 0 
            elif (labelmatrix[x][y] == most_frequent_value) & (image[x][y] > v):
                image[x][y] = 0 


    # cv.imshow('pls hoja', image)
    # cv.waitKey()
    # print(eqlist)
    return image

def CCAwithMode(image, v, layer): # v should be list
    KER, EPI, DRM, DEJ = [224, 224, 224], [160, 48, 112], [0, 255, 190], [255, 172, 255] #in BGR format
    shape = image.shape
    if len(shape) == 2:  # 2D array (grayscale image)
        height, width = shape
    elif len(shape) == 3:  # 3D array (color image)
        height, width, channels = shape
    else:
        raise ValueError("Unsupported number of dimensions in image")
    # z = np.array([0,0,0],dtype='uint8')
    resmatrix = np.full((height, width, 3),0,dtype='uint8')
    eqlist = []
    label = 0
    labelmatrix = np.zeros(image.shape, dtype='uint16')
    for x in range(0,height):
            for y in range(0,width):
                if image[x][y] in v:
                    neighbors = []
                    # taking current pixel and inserting its up and left labels in a list
                    if x > 0 and image[x-1][y] in v:
                        neighbors.append(labelmatrix[x-1][y])
                    if y > 0 and image[x][y-1] in v:
                        neighbors.append(labelmatrix[x][y-1])
                    if (y > 0 and x > 0) and image[x-1][y-1] in v:
                        neighbors.append(labelmatrix[x-1][y-1])
                    if (y < width-1 and x > 0):
                        if image[x-1][y+1] in v:
                            neighbors.append(labelmatrix[x-1][y+1])
    
                    if not neighbors: # if no neighbors in the list
                        label+=1
                        labelmatrix[x][y] = label
                        eqlist.append(label)
                    elif len(neighbors) == 1: # if only one label then copy that label on that coordinate
                        labelmatrix[x][y] = neighbors[0]
                    elif len(neighbors) > 1:  # if more than one label
                        if len(set(neighbors)) == 1: # if all having same label
                            labelmatrix[x][y] = neighbors[0] # assign that label
                        else:
                            labelmatrix[x][y] = max(neighbors)
                            labelmatrix[labelmatrix == min(neighbors)] = max(neighbors)

                    # resmatrix[x][y][:] = labelmatrix[x][y]
                    # if image[x][y] in v: 
                    #     resmatrix[x][y] = layer      
                        
    eqlist = np.delete(np.unique(labelmatrix), np.where(np.unique(labelmatrix) == 0))

    arr_flat = labelmatrix.flatten()

    unique_values, counts = np.unique(arr_flat, return_counts=True)
    sorted_indices = np.argsort(-counts)
    unique_values_sorted = unique_values[sorted_indices]
    max_count_index = np.argmax(counts)
    most_frequent_value = unique_values[max_count_index]
    second_most_frequent_value = None
    for val in unique_values_sorted:
        if val != unique_values[sorted_indices[0]]:
            second_most_frequent_value = val
            break

    for x in range(0,height):
        for y in range(0,width):
            if (labelmatrix[x][y] == second_most_frequent_value) & (image[x][y] in v):
                resmatrix[x][y] = layer 
    
    return resmatrix


def CCA(image, v, layer): # v should be list
    KER, EPI, DRM, DEJ = [224, 224, 224], [160, 48, 112], [0, 255, 190], [255, 172, 255] #in BGR format
    shape = image.shape
    if len(shape) == 2:  # 2D array (grayscale image)
        height, width = shape
    elif len(shape) == 3:  # 3D array (color image)
        height, width, channels = shape
    else:
        raise ValueError("Unsupported number of dimensions in image")
    # z = np.array([0,0,0],dtype='uint8')
    resmatrix = np.full((height, width, 3),0,dtype='uint8')
    eqlist = []
    label = 0
    labelmatrix = np.zeros(image.shape, dtype='uint16')
    for x in range(0,height):
            for y in range(0,width):
                if image[x][y] in v:
                    neighbors = []
                    # taking current pixel and inserting its up and left labels in a list
                    if x > 0 and image[x-1][y] in v:
                        neighbors.append(labelmatrix[x-1][y])
                    if y > 0 and image[x][y-1] in v:
                        neighbors.append(labelmatrix[x][y-1])
                    if (y > 0 and x > 0) and image[x-1][y-1] in v:
                        neighbors.append(labelmatrix[x-1][y-1])
                    if (y < width-1 and x > 0):
                        if image[x-1][y+1] in v:
                            neighbors.append(labelmatrix[x-1][y+1])
    
                    if not neighbors: # if no neighbors in the list
                        label+=1
                        labelmatrix[x][y] = label
                        eqlist.append(label)
                    elif len(neighbors) == 1: # if only one label then copy that label on that coordinate
                        labelmatrix[x][y] = neighbors[0]
                    elif len(neighbors) > 1:  # if more than one label
                        if len(set(neighbors)) == 1: # if all having same label
                            labelmatrix[x][y] = neighbors[0] # assign that label
                        else:
                            labelmatrix[x][y] = max(neighbors)
                            labelmatrix[labelmatrix == min(neighbors)] = max(neighbors)

                    resmatrix[x][y][:] = labelmatrix[x][y]
                    if image[x][y] in v: 
                        resmatrix[x][y] = layer      
                        
    eqlist = np.delete(np.unique(labelmatrix), np.where(np.unique(labelmatrix) == 0))

    arr_flat = labelmatrix.flatten()

    unique_values, counts = np.unique(arr_flat, return_counts=True)
    sorted_indices = np.argsort(-counts)
    unique_values_sorted = unique_values[sorted_indices]
    max_count_index = np.argmax(counts)
    most_frequent_value = unique_values[max_count_index]
    second_most_frequent_value = None
    for val in unique_values_sorted:
        if val != unique_values[sorted_indices[0]]:
            second_most_frequent_value = val
            break

    # for x in range(0,height):
    #     for y in range(0,width):
    #         if (labelmatrix[x][y] == most_frequent_value) & (image[x][y] in v):
    #             resmatrix[x][y] = layer 
    
    return resmatrix

# im_bw = cv.imread('Dataset/Train/Tissue/1.jpg', 0)
# # (thresh, im_bw) = cv.threshold(im_bw, 250, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# cv.imshow('input',im_bw)
# # im_bw = cv.bitwise_not(im_bw)
# # im_bw = [[0,255,0,0],
# #          [0,255,0,255],
# #          [255,255,0,255],
# #          [0,0,255,0]]

# im_bw = np.array(im_bw)
# cv.imshow('dsf',extractBackground(im_bw))
# cv.waitKey()