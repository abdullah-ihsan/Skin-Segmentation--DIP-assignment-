import cv2 as cv

def dice(filenum):
    maskimg = f'Dataset/Train/masks/{filenum}.png'
    outputimg = f'Dataset/Train/Output/{filenum}.png'

    img = cv.imread(outputimg,0)
    mask = cv.imread(maskimg,0)
    height, width = img.shape

    overlap = 0
    for x in range(0,height):
        for y in range(0,width):
            if img[x][y] == 0 or mask[x][y] == 0:
                continue
            if img[x][y] == mask[x][y]:
                overlap+=1

    area = height * width
    return overlap / area   # (2 * overlap) / (area of first set + area of second set) 