import os
import cv2
import copy
import numpy as np
import pandas as pd
import open3d as o3d

from utils import reverse, map_cleansing, MSRCR
from lama.bin import inpaint

def projection(pcdo):
    xyz = np.asarray(pcdo.points)
    rgb = np.asarray(pcdo.colors)

    margin=10
    minx=np.min(xyz[:,0])
    maxx=np.max(xyz[:,0])
    miny=np.min(xyz[:,1])
    maxy=np.max(xyz[:,1])
    dx=maxx-minx
    dy=maxy-miny

    ratio=dx/dy
    resx=int((len(xyz)*1/ratio)**0.5*ratio)
    resy=int((len(xyz)*1/ratio)**0.5)
    resolution=[resy,resx,3]

    if ratio>=(resolution[1]-2*margin)/(resolution[0]-2*margin):
        dp=dx/(resolution[1]-2*margin)
    else:
        dp=dy/(resolution[0]-2*margin)
    projectionimg=np.zeros(resolution)
    projectionimg=projectionimg.astype(np.uint8)

    label_list = np.zeros(projectionimg.shape[:2])
    for i in range(len(xyz)):
        lx = int((xyz[i][0]-minx)/dp)-1
        ly = int((maxy-xyz[i][1])/dp)-1
        if not label_list[ly, lx]:
            c=rgb[i]*255
            c=[c[2],c[1],c[0]]
            projectionimg[ly+margin][lx+margin] = c
            label_list[ly][lx] = 1
    projectionimg = cv2.flip(projectionimg, 1)

    return projectionimg


def imgInpaint(projectionimg):
    resolution = projectionimg.shape

    projectionimg=projectionimg.astype(np.uint8) 

    img_gray = cv2.cvtColor(projectionimg,cv2.COLOR_BGR2GRAY)
    maskindex=np.where(img_gray==0)
    mask=np.zeros(resolution[:-1])
    mask[maskindex]=255
    mask=mask.astype(np.uint8)

    projectionimg_float = projectionimg.astype('float32') / 255
    mask_float = mask.astype('float32') / 255
    dst = inpaint(projectionimg_float, mask_float)

    return dst

def illumination(dst):
    img=copy.deepcopy(dst)

    img_msrcr = MSRCR(img,sigma_list=[15, 80, 250],
                    G=192.0, b=-30.0,
                    alpha=125.0, beta=46.0,
                    low_clip=0.01, high_clip=0.8)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_r_img = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2HSV)

    v_r_img = hsv_r_img[:, :, 2] + 20
    v_r_img[v_r_img>255] = 255

    hsv_m_img1 = np.dstack((hsv_img[:, :, 0], hsv_img[:, :, 1], v_r_img))
    m_img1 = cv2.cvtColor(hsv_m_img1, cv2.COLOR_HSV2BGR)

    return img_msrcr, m_img1


def postprocess(pcd_ori, pcd_ds, pcdfi, point_map, ind_nan, **kwargs):
    print('postprocess begin')

    path = kwargs["output_path"]

    if not os.path.exists(path):
        os.makedirs(path)

    xyz0 = np.asarray(pcd_ori.points)
    rgb0 = np.asarray(pcd_ori.colors)
    xyz1 = np.asarray(pcd_ds.points)
    xyz2 = np.asarray(pcdfi.points)
    dis = xyz2 - xyz1

    for i in range(len(point_map)):
        xyz0[point_map[i]] += dis[i]
    ind = [point_map[i] for i in range(len(point_map)) if i not in ind_nan]
    ind = [j for i in ind for j in i]
    xyz0 = xyz0[ind]
    rgb0 = rgb0[ind]
    pcdo = o3d.geometry.PointCloud()
    pcdo.points = o3d.utility.Vector3dVector(xyz0)
    pcdo.colors = o3d.utility.Vector3dVector(rgb0)

    projectionimg = projection(pcdo)

    dst = imgInpaint(projectionimg)

    img_msrcr, m_img1 = illumination(dst)

    # cv2.imwrite(path+'/output.jpg',dst)
    o3d.io.write_point_cloud(path+'/result.ply',pcdo,write_ascii= True)

    print('postprocess done')

    return pcdo, dst, m_img1
