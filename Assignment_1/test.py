import numpy as np
import cv2 as cv
import random


# # im_bw = cv.imread('2.[8192x512].jpg')
# # cv.imshow('input',im_bw)
# # cv.waitKey()
# # print(im_bw)

# def cca(image):
#     v = 255
#     height, width = image.shape
#     eqlist = []
#     label = 0
#     labelmatrix = np.zeros(image.shape)
#     for x in range(0,height):
#         for y in range(0,width):
#             if image[x][y] == v:
#                 neighbors = []
#                 if x > 0 and image[x-1][y] == v:
#                     neighbors.append(labelmatrix[x-1][y])
#                 if y > 0 and image[x][y-1] == v:
#                     neighbors.append(labelmatrix[x][y-1])
#                 if not neighbors:
#                     label+=1
#                     labelmatrix[x][y] = label
#                     eqlist.append(label)
#                 elif len(neighbors) == 1:
#                     labelmatrix[x][y] = neighbors[0]
#                 elif len(neighbors) == 2:
#                 # if all having same label
#                     if neighbors[0] == neighbors[1]:
#                         # assign that label
#                         labelmatrix[x][y] = neighbors[0]
#                     else:
#                         # assign label based on priority (larger label has higher priority)
#                         labelmatrix[x][y] = max(neighbors)
#                         labelmatrix[labelmatrix == min(neighbors)] = max(neighbors)
#     eqlist = np.delete(np.unique(labelmatrix), np.where(np.unique(labelmatrix) == 0))
#     return len(eqlist)

# image = cv.imread('RA23-01882-A1-1-PAS.[1536x2560].jpg',0)
# (thresh, im_bw) = cv.threshold(image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# cv.imshow('dsfa',im_bw)
# cv.waitKey()
# # im_bw = [[0,255,0,0],
# # [0,255,0,0],
# # [255,255,0,255],
# # # [0,0,255,0]]
# # im_bw = np.array(im_bw)
# print(cca(im_bw))

#################################################################

# import numpy as np

# # Example 2D NumPy array
# arr_2d = np.array([[1, 2, 3],
#                    [1, 2, 3],
#                    [1, 1, 3]])

# # Flatten the array
# arr_flat = arr_2d.flatten()

# # Get unique values and their counts
# unique_values, counts = np.unique(arr_flat, return_counts=True)

# # Sort unique values by counts in descending order
# sorted_indices = np.argsort(-counts)
# unique_values_sorted = unique_values[sorted_indices]

# # Second most frequent value
# second_most_frequent_value = None

# # Loop through sorted unique values to find the second most frequent value
# for val in unique_values_sorted:
#     if val != unique_values[sorted_indices[0]]:
#         second_most_frequent_value = val
#         break

# print("Second most frequent value:", second_most_frequent_value)


###########################################################################
import time

def cca(image):
    start_time = time.time()
    v = 251
    height, width = image.shape
    eqlist = []
    label = 0
    labelmatrix = np.zeros(image.shape, dtype='uint8')
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
    
                    # if labelmatrix[x][y] > 0: labelmatrix[x][y] = 255                    
        # print(labelmatrix)
    cv.imshow('segmented',labelmatrix)
    # cv.waitKey()
    eqlist = np.delete(np.unique(labelmatrix), np.where(np.unique(labelmatrix) == 0))
    print("--- %s seconds ---" % (time.time() - start_time))
    return len(eqlist)

image = cv.imread('part-b.png',0)
(thresh, im_bw) = cv.threshold(image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow('dsfa',im_bw)
cv.waitKey()
print(cca(im_bw))