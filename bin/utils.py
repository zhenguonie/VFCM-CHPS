import math
import cv2
import numpy as np
from tqdm import trange

def dcmp(x):
    if abs(x) < 1e-6:
        return 0
    else:
        return -1 if x<0 else 1

def onSegment(p1, p2, q):
    t1 = p1-q
    t2 = p2-q
    crossProduct = t1[0]*t2[1] - t1[1]*t2[0]
    product = t1[0]*t2[0] + t1[1]*t2[1]
    return dcmp(crossProduct) == 0 and dcmp(product) <= 0

def inPolygon(polygon, p):
    Flag = np.zeros([p.shape[0]])
    for k in trange(p.shape[0]):
        flag = False
        lines = zip(range(len(polygon)), range(1, len(polygon)+1))
        for i, j in lines:
            if j == len(polygon):
                j = 0
            p1 = np.array(polygon[i])
            p2 = np.array(polygon[j])
            if onSegment(p1, p2, p[k,:]):
                Flag[k] = True
                break
            if ((p1[1]>p[k,1]) != (p2[1]>p[k,1])) and (p[k,0] < (p[k,1]-p1[1])*(p1[0]-p2[0])/(p1[1]-p2[1])+p1[0]):
                flag = not flag
        Flag[k] = flag
    return Flag

def threshold(img, th):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])[1:]
    v1=0
    v2=255
    p1=0
    p2=0
    for i in range(th):
        if hist[i] > p1:
            v1 = i
            p1 = hist[i]
    for j in range(254, th, -1):
        if hist[j] > p2:
            v2 = j
            p2 = hist[j]
    pth = int((v1*v2)**0.5)
    return pth

def reverse(pcd, ind_nan):
    n = np.asarray(pcd.points).shape[0]
    ind = []
    for i in range(n):
        if i not in ind_nan:
            ind.append(i)
    ind = np.array(ind)
    return ind

def map_cleansing(map, ind_nan):
    for i in range(len(map)):
        map[i] = [int(j) for j in map[i] if not math.isnan(j)]
    return map

def map_flatten(map):
    fmap = []
    for i in range(len(map)):
        inv = np.asarray(map[i])
        for j in range(len(inv)):
            fmap.append(inv[j])
    fmap = np.sort(np.asarray(fmap, dtype=int))
    return fmap

def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex

def multiScaleRetinex(img, sigma_list):

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex

def colorRestoration(img, alpha, beta):

    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration

def simplestColorBalance(img, low_clip, high_clip):    

    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):            
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
                
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img    

def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):

    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)    
    img_color = colorRestoration(img, alpha, beta)    
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255
    
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)       

    return img_msrcr