import numpy as np
import open3d as o3d
import pandas as pd
import copy
import math
import cv2
import os
from copy import deepcopy
from plyfile import *

from utils import inPolygon, threshold
from polygonal_approximation import thick_polygonal_approximate

def load_pcd(filename):
    '''loading pcd'''

    pcdo=o3d.geometry.TriangleMesh()

    plydata = PlyData.read(filename)
    if 'face' in plydata:
        if len(plydata['face']['vertex_indices'])>0:
            pcdo=o3d.geometry.TriangleMesh()
            plytype='trianglemesh'
            mesh=plydata['face']['vertex_indices']
            pcdo.triangles=o3d.utility.Vector3iVector(mesh)
        else:
            pcdo=o3d.geometry.PointCloud()
            plytype='pointcloud'
    else:
        pcdo=o3d.geometry.PointCloud()
        plytype='pointcloud'
    xlist = plydata['vertex']['x']
    ylist = plydata['vertex']['y']
    zlist = plydata['vertex']['z']
    xyzo=np.array([xlist,ylist,zlist])
    xyzo=xyzo.transpose()
    mean=np.mean(xyzo,axis=0)
    xyzo=xyzo-mean

    if plytype=='pointcloud':
        rlist = plydata['vertex']['red']
        glist = plydata['vertex']['green']
        blist = plydata['vertex']['blue']
        rgbo=np.array([rlist,glist,blist])
        rgbo=rgbo.transpose()/255
        pcdo.colors=o3d.utility.Vector3dVector(rgbo)
        pcdo.points=o3d.utility.Vector3dVector(xyzo)
    else:
        pcdo.vertices=o3d.utility.Vector3dVector(xyzo)

    return pcdo, plytype

def rigid_transformation(pcdo):
    '''rigid transformation'''

    xyzo = np.asarray(pcdo.points)

    # fitting plane, rotation
    X=copy.deepcopy(xyzo)
    X[:,2]=1
    Z=xyzo[:,2]
            
    A=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Z)
    a=A[0]
    b=A[1]

    axis=np.array([[b/(a**2+b**2)**0.5],[-a/(a**2+b**2)**0.5],[0]])
    angle=-1*math.acos(1/(a**2+b**2+1)**0.5)
    aa=np.dot(axis,angle)

    pcd1=copy.deepcopy(pcdo)
    R=pcd1.get_rotation_matrix_from_axis_angle(aa)
    pcd1.rotate(R,center=(0,0,0))

    return pcd1

def projection(pcd1, plytype, margin):
    '''projection'''

    if plytype=='trianglemesh':
        resolution=[300,400]
        xyz1=np.asarray(pcd1.vertices)
    else:
        resolution=[300,400,3]
        xyz1=np.asarray(pcd1.points)
        rgb1=np.asarray(pcd1.colors)
    minx=np.min(xyz1[:,0])
    maxx=np.max(xyz1[:,0])
    miny=np.min(xyz1[:,1])
    maxy=np.max(xyz1[:,1])
    mm = np.asarray([[minx, maxx], [miny, maxy]])
    dx=maxx-minx
    dy=maxy-miny
    if dx/dy>=(resolution[1]-2*margin)/(resolution[0]-2*margin):
        dp=dx/(resolution[1]-2*margin)
    else:
        dp=dy/(resolution[0]-2*margin)
    projectionimg=np.zeros(resolution)
    projectionimg=projectionimg.astype(np.uint8)

    return projectionimg, mm, dp

def edge_extraction(path, pcd1, plytype, projectionimg, mm, dp, margin):
    '''extract edges'''

    minx = mm[0, 0]
    maxy = mm[1, 1]
    resolution = projectionimg.shape

    if plytype=='trianglemesh':
        xyz1=np.asarray(pcd1.vertices)
        edge_manifold_boundary = pcd1.is_edge_manifold(allow_boundary_edges=False)
        if not edge_manifold_boundary:
            edges = pcd1.get_non_manifold_edges(allow_boundary_edges=False)
            
            edges_arr=np.asarray(edges)
            coords_arr=[]
            for i in range(len(edges_arr)):
                e=edges_arr[i]
                pixelx1=int((xyz1[e[0]][0]-minx)/dp)-1+margin
                pixely1=int((maxy-xyz1[e[0]][1])/dp)-1+margin
                pixelx2=int((xyz1[e[1]][0]-minx)/dp)-1+margin
                pixely2=int((maxy-xyz1[e[1]][1])/dp)-1+margin
                coords=[pixelx1,pixely1,pixelx2,pixely2]
                coords_arr.append(coords)
            coords_arr=np.asarray(coords_arr)
            for x1, y1, x2, y2 in coords_arr:
                cv2.line(projectionimg, (x1, y1), (x2, y2), 255, 2)
                cv2.imwrite(path+'/temp.png',projectionimg)
                
    elif plytype=='pointcloud':
        xyz1=np.asarray(pcd1.points)
        rgbo=np.asarray(pcd1.colors)
        
        label_list = np.zeros(projectionimg.shape[:2])
        for i in range(len(xyz1)):
            lx = int((xyz1[i][0]-minx)/dp)-1
            ly = int((maxy-xyz1[i][1])/dp)-1
            if not label_list[lx, ly]:
                projectionimg[ly+margin][lx+margin] = rgbo[i]*255
                label_list[lx][ly] = 1
        
        projectionimg=projectionimg.astype(np.uint8)        
        img_gray=cv2.cvtColor(projectionimg,cv2.COLOR_BGR2GRAY)
        maskindex=np.where(img_gray==0)
        mask=np.zeros(resolution[0:2])
        mask[maskindex]=255
        mask=mask.astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        dst = cv2.morphologyEx(projectionimg, cv2.MORPH_CLOSE, kernel, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, 1)
        img1 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        
        canny = cv2.Canny(img1, 50, 150)
        canny2=copy.deepcopy(canny)
        lines = cv2.HoughLinesP(canny2,1,np.pi/180,50,minLineLength=30,maxLineGap=100)
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(canny2, (x1, y1), (x2, y2), 255, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        projectionimg2 = cv2.morphologyEx(canny, cv2.MORPH_CLOSE,kernel)
    
    img2 = cv2.cvtColor(projectionimg,cv2.COLOR_BGR2GRAY)

    th = threshold(img2, 127)
    _,img5 = cv2.threshold(img2, th, 255, cv2.THRESH_TOZERO)

    contours5, hierarchy = cv2.findContours(img5,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    projectionimg12 = copy.deepcopy(projectionimg)
    cv2.drawContours(projectionimg12, contours5, -1, (0, 0, 255), 1)

    la=[]
    for i in range(len(contours5)):
        area=cv2.contourArea(contours5[i])
        la.append([i,area])
    la=sorted(la,key=lambda x:x[1],reverse=True)
    cnt3=contours5[la[0][0]]

    projectionimg15 = copy.deepcopy(projectionimg)
    cv2.drawContours(projectionimg15, cnt3, -1, (0, 0, 255), 1)

    epsilon3 = 0.01*cv2.arcLength(cnt3, True)
    cnt4 = np.reshape(cnt3, (cnt3.shape[0], cnt3.shape[2]))
    approx5 = thick_polygonal_approximate(cnt4, epsilon3)
    projectionimg14 = copy.deepcopy(projectionimg)
    cv2.drawContours(projectionimg14, [approx5], -1, (0, 0, 255), 1)

    rect2 = cv2.minAreaRect(cnt3)
    box2 = cv2.boxPoints(rect2)  
    box2 = np.int0(box2)
    V2=box2
    projectionimg14 = copy.deepcopy(projectionimg)
    cv2.drawContours(projectionimg14,[V2],-1,[255,0,0],3)

    return V2, approx5

def pixel2coords(pcd1, approx, V, mm, dp, margin):
    '''pixels to coords'''

    minx = mm[0, 0]
    maxy = mm[1, 1]

    Vertex_app = np.array([(approx[:, 0]+1-margin)*dp+minx, maxy-(approx[:, 1]+1-margin)*dp, np.zeros(approx.shape[0])]).transpose()
    pcdva=o3d.geometry.PointCloud()
    pcdva.points=o3d.utility.Vector3dVector(Vertex_app)

    midx=0
    midy=0
    for i in range(4):
        midx+=V[i][0]/4
        midy+=V[i][1]/4
    midx=(midx+1-margin)*dp+minx
    midy=maxy-(midy+1-margin)*dp
    Vertex=[[],[],[],[]]
    for v in V:
        t=[(v[0]+1-margin)*dp+minx, maxy-(v[1]+1-margin)*dp, 0]
        if t[0]<=midx and t[1]>midy:
            Vertex[0]=t
        elif t[0]>midx and t[1]>midy:
            Vertex[1]=t
        elif t[0]>midx and t[1]<=midy:
            Vertex[2]=t
        elif t[0]<=midx and t[1]<=midy:
            Vertex[3]=t
    pcdv=o3d.geometry.PointCloud()
    pcdv.points=o3d.utility.Vector3dVector(np.array(Vertex))

    ti=[]
    for i in range(4):
        direction=[Vertex[i%4][0]-Vertex[(i+1)%4][0],Vertex[i%4][1]-Vertex[(i+1)%4][1]]
        if direction[0]!=0:
            t=-1*direction[1]/direction[0]
        else:
            t=float('inf')
        ti.append(t)

    theta=math.atan(ti[0])

    pcd2=o3d.geometry.PointCloud()
    xyz1=np.asarray(pcd1.points)
    rgb1=np.asarray(pcd1.colors)
    pcd2.points=o3d.utility.Vector3dVector(xyz1)
    pcd2.colors=o3d.utility.Vector3dVector(rgb1)

    R1=pcd2.get_rotation_matrix_from_zyx((theta,0,0))
    pcd2.rotate(R1,center=(0,0,0))

    xyz2=np.asarray(pcd2.points)

    R2=pcdv.get_rotation_matrix_from_zyx((theta,0,0))
    pcdv.rotate(R2,center=(0,0,0))
    Vertex2=np.asarray(pcdv.points)[:,:2]

    R3=pcdva.get_rotation_matrix_from_zyx((theta,0,0))
    pcdva.rotate(R3,center=(0,0,0))
    Vertex_app2=np.asarray(pcdva.points)[:,:2]

    # point cloud sampling
    pcd3 = deepcopy(pcd2)
    box = pcd3.get_axis_aligned_bounding_box()
    boxMin = box.get_min_bound()
    boxMax = box.get_max_bound()
    diagonal = np.sum(np.power(boxMax-boxMin, 2))**0.5
    downpcd, _, point_map = pcd3.voxel_down_sample_and_trace(voxel_size=diagonal/250, min_bound=boxMin, max_bound=boxMax)
    xyzd = np.asarray(downpcd.points)
    rgbd = np.asarray(downpcd.colors)

    # Determine whether the point is inside the rectangular box and remove outside
    margin=0
    ind1=np.where((xyzd[:,0]>min(Vertex2[:,0])+margin) & (xyzd[:,0]<max(Vertex2[:,0])-margin))
    temp=xyzd[ind1]
    tempc=rgbd[ind1]
    point_map = [point_map[i] for i in ind1[0].tolist()]
    ind2=np.where((temp[:,1]>min(Vertex2[:,1])+margin) & (temp[:,1]<max(Vertex2[:,1])-margin))
    xyz3=temp[ind2]
    rgb3=tempc[ind2]
    point_map = [point_map[i] for i in ind2[0].tolist()]

    ind = np.where(inPolygon(Vertex_app2, xyz3[:, :2]))
    xyz4 = xyz3[ind]
    rgb4 = rgb3[ind]
    scalefactor=(np.max(xyz4,axis=0)[0]-np.min(xyz4,axis=0)[0])/100
    xyz4=xyz4/scalefactor
    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(xyz4)
    pcd4.colors = o3d.utility.Vector3dVector(rgb4)
    point_map = [point_map[i] for i in ind[0].tolist()]

    pcd5, ind3 = pcd4.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    point_map2 = [point_map[i] for i in ind3]

    xyz2 = np.asarray(pcd2.points)
    rgb2 = np.asarray(pcd2.colors)
    xyz2=xyz2/scalefactor
    pcd2.points = o3d.utility.Vector3dVector(xyz2)
    pcd2.colors = o3d.utility.Vector3dVector(rgb2)

    return pcd2, pcd5, point_map2

def save(path, pcd_ori, pcd_ds, point_map):
    '''saving'''

    o3d.io.write_point_cloud(path+'/interpcd_ori.ply',pcd_ori,write_ascii= True)
    o3d.io.write_point_cloud(path+'/interpcd_downsample.ply',pcd_ds,write_ascii= True)

    maps = pd.DataFrame(data=point_map, index=None, columns=None)
    maps.to_csv(path+'/point_map.csv')

    # o3d.visualization.draw_geometries([pcd_ori],mesh_show_wireframe=True,mesh_show_back_face=True)
    # o3d.visualization.draw_geometries([pcd_ds],mesh_show_wireframe=True,mesh_show_back_face=True)

def preprocess(**kwargs):
    print('preprocessing begin')

    filename = kwargs["input_pcd"]
    path = kwargs["work_path"]

    if not os.path.exists(path):
        os.makedirs(path)

    margin=50

    pcdo, plytype = load_pcd(filename)
    pcd1 = rigid_transformation(pcdo)
    projectionimg, mm, dp = projection(pcd1, plytype, margin)
    V, approx = edge_extraction(path, pcd1, plytype, projectionimg, mm, dp, margin)
    pcd_ori, pcd_ds, point_map = pixel2coords(pcd1, approx, V, mm, dp, margin)
    save(path, pcd_ori, pcd_ds, point_map)

    print('preprocessing done')

    return pcd_ds, pcd_ori, point_map
