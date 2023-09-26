import psutil
import os
import time
import copy

import trimesh
import numpy as np
import taichi as ti
import open3d as o3d
from tqdm import trange
import pandas as pd
import pyvista as pv

# if gpu is unavailable, then toggle the following lines' comments.
arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch)
# ti.init(arch=ti.cpu)

def load_mesh(pcdnm):
    global substeps, n_vertexs, vertexss, faces, max_cols, max_cols_hold
    global xin, x, v, xfin, xp, vp, top, n_adface, pi, floor, colors
    global neighbor, neighbor_hold, neighbor_hold_l, adjacency_faces, adjacency_uns_v, adjacency_theta
    global face_v, neighbor_arr, neighbors_hold_arr, neighbors_hold_len_arr, adjacency, adjacency_uns, neighbors_theta_id
    global adjacency_angles, adjacency_force_i, adjacency_force_j, force, x_delta, delta_id
    global spring_Y, dashpot_damping, drag_damping, dt, rho, rho_ini, scale_l, mesh_scale
   
    plydata = o3d.geometry.PointCloud()
    plydata = o3d.geometry.PointCloud(pcdnm)

    pi = np.pi

    mesh_scale = 2

    # # Estimate normals for the point cloud
    plydata.estimate_normals()

    # Estimate radius for rolling ball
    distances = plydata.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    avg_dist1 = np.min(distances)
    avg_dist2 = np.max(distances)
    radius = np.linspace(avg_dist1, avg_dist2,5)

    # Create a mesh from the point cloud using ball pivoting
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        plydata,
        o3d.utility.DoubleVector(radius)
        )
    
    mesh = trimesh.Trimesh(vertices = np.asarray(mesh_o3d.vertices),
                            faces = np.asarray(mesh_o3d.triangles),
                            vertex_normals = np.asarray(mesh_o3d.vertex_normals),
                            vertex_colors = np.asarray(mesh_o3d.vertex_colors),
                            process=False)

    colors = mesh.visual.vertex_colors
#################################################################################
    mesh_BPA = pv.wrap(mesh)
    mesh_de2d = mesh_BPA.delaunay_2d()

    mesh_con = pv.wrap(mesh)
    filled_mesh = mesh_con.connectivity()
    region_id = filled_mesh['RegionId']
    region_id = np.logical_not(np.not_equal(region_id, 0))
    mesh.faces = mesh.faces[region_id]

    mesh_o3d_ids = set(np.ndarray.flatten(mesh.faces))
    ori_ids = set(range(len(mesh.vertices)))
    rest_ods = np.asarray(list(ori_ids.difference(mesh_o3d_ids)))



    de2d_faces = mesh_de2d.faces
    de2d_faces = de2d_faces.reshape(int(len(mesh_de2d.faces)/4),4)
    mesh_d2d = trimesh.Trimesh(vertices = np.asarray(mesh_o3d.vertices),
                            faces = de2d_faces[:,1:4],
                            vertex_normals = np.asarray(mesh_o3d.vertex_normals),
                            vertex_colors = np.asarray(mesh_o3d.vertex_colors),
                            process=False)
    mean_area = np.mean(mesh_d2d.area_faces)
    mask1 = mesh_d2d.area_faces > (mean_area * mesh_scale)

    face_normals = mesh_d2d.face_normals
    angles = np.arccos(face_normals.dot([0,0,1]))*180/np.pi
    min_angles = min(angles)
    mask3 = np.abs(angles-90) < 5

    mask = np.logical_or(mask1,mask3)
    mesh_d2d.faces = mesh_d2d.faces[~mask]

    cell_extract = mesh_d2d.vertex_faces[rest_ods]
    cell_extract_set = set(cell_extract.flatten())
    cell_extract_set.remove(-1)
    
    mesh_pv_add = mesh_d2d.faces[np.asarray(list(cell_extract_set))]
    faces = np.append(mesh.faces, mesh_pv_add, axis = 0)
    mesh.faces = faces

    mesh_con = pv.wrap(mesh)
    filled_mesh = mesh_con.connectivity()
    region_id = filled_mesh['RegionId']
    region_id = np.logical_not(np.not_equal(region_id, 0))
    mesh.faces = mesh.faces[region_id]


    
######################################################################################

    vertexss = mesh.vertices
    
    length_pc = max(max(vertexss[:,0]) - min(vertexss[:,0]), max(vertexss[:,1]) - min(vertexss[:,1])) 
    length_scale = length_pc // 10
    if length_scale == 0:
        length_scale = 1

    scale_l = 0.01 / length_scale
    
    floor_s = min(vertexss[:,2])-1
    floor = min(vertexss[:,2])*scale_l
    floor2 = 0.0
    top = max(vertexss[:,2])*scale_l

        
    n_vertexs = len(vertexss)
    lengths = mesh.edges_unique_length

    len_unique = mesh.edges_unique

    last_id = np.arange(0,len(vertexss))
    length = len(vertexss)
########################################################################################################

    neighbors = {p: [] for p in range(len(vertexss))}
    for edge in mesh.edges_unique:
        p1, p2 = edge
        neighbors[p1].append(p2)
        neighbors[p2].append(p1)

    max_cols = max(len(row) for row in neighbors.values())

    # create empty array
    neighbor_arr = np.zeros((n_vertexs, max_cols), dtype=np.int32)

    # fill array
    for i, row in enumerate(neighbors.values()):
        for j, value in enumerate(row):
            neighbor_arr[i, j] = value
        if len(row) < max_cols:
            neighbor_arr[i, len(row):] = -1
        
        adjacency = mesh.face_adjacency

    area_faces = mesh.area_faces
    adjacency_uns = mesh.face_adjacency_unshared
    adjacency_edges = mesh.face_adjacency_edges
    edges_unique_i = mesh.edges_unique_inverse
    n_adface = len(adjacency)
##############################################

    index = 0
    neighbors_hold_length = {p: [] for p in range(len(vertexss))}
    neighbors_hold = {p: [] for p in range(len(vertexss))}
    neighbors_theta = {p: [] for p in range(len(vertexss))}
    for face in adjacency:
        p1, p2 = face
        S = area_faces[p1] + area_faces[p2]
        aa, cc = adjacency_edges[index]
        q1, q2 = adjacency_uns[index]
    
        m = np.linalg.norm(vertexss[aa] - vertexss[cc])
        a = vertexss[aa] - vertexss[q1]
        c = vertexss[aa] - vertexss[q2]
        b = vertexss[cc] - vertexss[q1]
        d = vertexss[cc] - vertexss[q2]

    
        D2 = (np.dot(a.T, a) - np.dot(b.T, b) + np.dot(c.T, c) - np.dot(d.T, d))**2
    
        L = 0.5 * np.sqrt(16 * S**2 + D2) / m
        neighbors_hold_length[q1].append(L)
        neighbors_hold_length[q2].append(L)
        neighbors_hold[q1].append(q2)
        neighbors_hold[q2].append(q1)
        neighbors_theta[q1].append(index)
        neighbors_theta[q2].append(index)
        index +=1

    max_cols_hold = max(len(row) for row in neighbors_hold.values())

    # create empty array
    neighbors_hold_arr = np.zeros((n_vertexs, max_cols_hold), dtype=np.int32)

    # fill array
    for i, row in enumerate(neighbors_hold.values()):
        for j, value in enumerate(row):
            neighbors_hold_arr[i, j] = value
        if len(row) < max_cols_hold:
            neighbors_hold_arr[i, len(row):] = -1
        
    # create empty array
    neighbors_hold_len_arr = np.zeros((n_vertexs, max_cols_hold), dtype=np.float32)

    # fill array
    for i, row in enumerate(neighbors_hold_length.values()):
        for j, value in enumerate(row):
            neighbors_hold_len_arr[i, j] = value
        if len(row) < max_cols_hold:
            neighbors_hold_len_arr[i, len(row):] = -1


    neighbors_theta_id = np.zeros((n_vertexs, max_cols_hold), dtype=np.float32)

    for i, row in enumerate(neighbors_theta.values()):
        for j, value in enumerate(row):
            neighbors_theta_id[i, j] = int(value)
        if len(row) < max_cols_hold:
            neighbors_theta_id[i, len(row):] = -1

    dt = 0.00001
    substeps = 500

    gravity = ti.Vector([0, 0, 0])
    spring_Y = 800
    dashpot_damping = 3e5
    drag_damping = 0.01

    xin = ti.Vector.field(3, dtype=float, shape=(n_vertexs))
    xfin = ti.Vector.field(3, dtype=float, shape=(n_vertexs))

    x = ti.Vector.field(3, dtype=float, shape=(n_vertexs))
    v = ti.Vector.field(3, dtype=float, shape=(n_vertexs))
    force = ti.Vector.field(3, dtype=float, shape=(n_vertexs))

    xp = ti.Vector.field(3, dtype=float, shape=(1, ))
    vp = ti.Vector.field(3, dtype=float, shape=(1, ))

    delta_id = ti.field(dtype=int, shape=(n_vertexs))

    neighbor = ti.Vector.field(int(max_cols), dtype=int, shape=(n_vertexs))
    neighbor_hold = ti.Vector.field(int(max_cols_hold), dtype=int, shape=(n_vertexs))
    neighbor_hold_l = ti.Vector.field(int(max_cols_hold), dtype=float, shape=(n_vertexs))

    adjacency_faces = ti.Vector.field(2, dtype=int, shape=(n_adface))
    adjacency_uns_v = ti.Vector.field(2, dtype=int, shape=(n_adface))
    adjacency_force_i = ti.Vector.field(3, dtype=int, shape=(n_adface))
    adjacency_force_j = ti.Vector.field(3, dtype=int, shape=(n_adface))
    adjacency_angles = ti.field(dtype=float, shape=(n_adface))
    adjacency_theta = ti.Vector.field(int(max_cols_hold), dtype=int, shape=(n_vertexs))
    face_v = ti.Vector.field(3, dtype=int, shape=(len(faces)))

    rho_ini = ti.field(dtype=float, shape=(n_vertexs))
    rho = ti.field(dtype=float, shape=(n_vertexs))
    x_delta = ti.Vector.field(3, dtype=float, shape=(n_vertexs))

def initialize_mass_points():
    xin.from_numpy(vertexss*scale_l)
    x.from_numpy(vertexss*scale_l)
    neighbor.from_numpy(neighbor_arr)
    neighbor_hold.from_numpy(neighbors_hold_arr)
    neighbor_hold_l.from_numpy(neighbors_hold_len_arr*scale_l)
    adjacency_faces.from_numpy(adjacency)
    adjacency_uns_v.from_numpy(adjacency_uns)

    adjacency_theta.from_numpy(neighbors_theta_id)

    face_v.from_numpy(faces)
    xp[0][2] = top        

@ti.kernel
def rhoini():
    
    n_z = ti.Vector([0.0, 0.0, 1.0]) 
        
    for i in ti.ndrange(n_adface):

        f1 = adjacency_faces[i][0]
        f2 = adjacency_faces[i][1]
        p1 = adjacency_uns_v[i][0]
        p2 = adjacency_uns_v[i][1]

        i_1 = x[face_v[f1][0]]
        j_1 = x[face_v[f2][0]]
        i_2 = x[face_v[f1][1]]
        j_2 = x[face_v[f2][1]]
        i_3 = x[face_v[f1][2]]
        j_3 = x[face_v[f2][2]]
        v_1 = ti.math.cross((i_3 - i_1), (i_3 - i_2))
        v_2 = ti.math.cross((j_3 - j_1), (j_3 - j_2))
                
        if v_1[2] < 0:
            v_1 = v_1 * -1
                    
        if v_2[2] < 0:
            v_2 = v_2 * -1

        the = ti.math.dot(v_1, v_2) / (v_1.norm() * v_2.norm())
        if the > 1:
            the = 1
        adjacency_angles[i] = ti.acos(the)

        n_1 = x[p1] - (i_1 + i_2 + i_3) / 3                
        alpha_1 = ti.acos(ti.math.dot(n_1, n_z) / n_1.norm())
        w1 = -1
        if alpha_1 > 0.5*pi:
            alpha_1 = alpha_1 - 0.5*pi
            w1 = 1     


        n_2 = x[p2] - (j_1 + j_2 + j_3) / 3
        alpha_2 = ti.acos(ti.math.dot(n_2, n_z) / n_2.norm())
        w2 = -1                
        if alpha_2 > 0.5*pi:
            alpha_2 = alpha_2 - 0.5*pi
            w2 = 1
                
                    
        d1 = v_1.normalized()
        d2 = v_2.normalized()
                
        W1 = alpha_1 / (alpha_1 + alpha_2) * w1
        W2 = alpha_2 / (alpha_1 + alpha_2) * w2
            
        adjacency_force_i[i] = W1 * d1
        adjacency_force_j[i] = W2 * d2
            
        
        
    for i in ti.ndrange(n_vertexs):
        rho_in = 0.0
        rho_i = 0.0
                
        for k in ti.static(range(max_cols_hold)):            
            j = neighbor_hold[i][k]
            q = adjacency_theta[i][k]
            if j != -1:
                theta = adjacency_angles[q]

                if 0.5236 < theta < 2.6179:  
                    force[i] += 1 * adjacency_force_i[q] / ti.cos(0.4 * (pi - theta)) ** 2                    
                    force[j] += 1 * adjacency_force_j[q] / ti.cos(0.4 * (pi - theta)) ** 2 
                    
                    
                x_ij = x[i] - x[j]
                x_ij[2] = 0
                v_ij = v[i] - v[j]
                original_dist = neighbor_hold_l[i][k] * 1
                d = x_ij.normalized()
                current_dist = x_ij.norm()

                # Spring force
                force[i] += -0.5*spring_Y * d * (current_dist / original_dist - 1) 
                force[j] += 0.5*spring_Y * d * (current_dist / original_dist - 1) 
                # # # Dashpot damping
                force[i] += -v_ij.dot(d) * d * dashpot_damping * drag_damping   
                force[j] += v_ij.dot(d) * d * dashpot_damping * drag_damping 
                

        for k in ti.static(range(max_cols)):            
            j = neighbor[i][k]
            if j != -1:
                x_ij = x[i] - x[j]
                
                x_ij[2] = 0
                v_ij = v[i] - v[j]
                ori_ij = xin[i] - xin[j]
                d = x_ij.normalized()
                                                
                original_dist = ori_ij.norm() * 1
                current_dist = x_ij.norm()

                # Spring force
                force[i] += -0.5*spring_Y * d * (current_dist / original_dist - 1) 
                force[j] += 0.5*spring_Y * d * (current_dist / original_dist - 1) 
                # # # Dashpot damping
                force[i] += -v_ij.dot(d) * d * dashpot_damping * drag_damping   
                force[j] += v_ij.dot(d) * d * dashpot_damping * drag_damping 
                
                rho_i += current_dist
                rho_in += original_dist
                
        rho[i] = rho_i
        rho_ini[i] = rho_in


    for i  in ti.ndrange(n_vertexs):                
        v[i] += force[i] * dt * rho_ini[i] / rho[i]    
        x[i] += dt * v[i]
        force[i] = ti.Vector([0.0, 0.0, 0.0])     


@ti.kernel
def substep():
    
    n_z = ti.Vector([0.0, 0.0, 1.0]) 
        
    for i in ti.ndrange(n_adface):

        f1 = adjacency_faces[i][0]
        f2 = adjacency_faces[i][1]
        p1 = adjacency_uns_v[i][0]
        p2 = adjacency_uns_v[i][1]

        i_1 = x[face_v[f1][0]]
        j_1 = x[face_v[f2][0]]
        i_2 = x[face_v[f1][1]]
        j_2 = x[face_v[f2][1]]
        i_3 = x[face_v[f1][2]]
        j_3 = x[face_v[f2][2]]
        v_1 = ti.math.cross((i_3 - i_1), (i_3 - i_2))
        v_2 = ti.math.cross((j_3 - j_1), (j_3 - j_2))
                
        if v_1[2] < 0:
            v_1 = v_1 * -1
                    
        if v_2[2] < 0:
            v_2 = v_2 * -1

        the = ti.math.dot(v_1, v_2) / (v_1.norm() * v_2.norm())
        if the > 1:
            the = 1
        adjacency_angles[i] = ti.acos(the)

        n_1 = x[p1] - (i_1 + i_2 + i_3) / 3                
        alpha_1 = ti.acos(ti.math.dot(n_1, n_z) / n_1.norm())
        w1 = -1
        if alpha_1 > 0.5*pi:
            alpha_1 = alpha_1 - 0.5*pi
            w1 = 1     

        n_2 = x[p2] - (j_1 + j_2 + j_3) / 3
        alpha_2 = ti.acos(ti.math.dot(n_2, n_z) / n_2.norm())
        w2 = -1                
        if alpha_2 > 0.5*pi:
            alpha_2 = alpha_2 - 0.5*pi
            w2 = 1


        d1 = v_1.normalized()
        d2 = v_2.normalized()
                
        W1 = alpha_1 / (alpha_1 + alpha_2) * w1
        W2 = alpha_2 / (alpha_1 + alpha_2) * w2
            
        adjacency_force_i[i] = W1 * d1
        adjacency_force_j[i] = W2 * d2

        
    for i in ti.ndrange(n_vertexs):
        
        rho_i = 0.0
                
        for k in ti.static(range(max_cols_hold)):            
            j = neighbor_hold[i][k]
            q = adjacency_theta[i][k]
            if j != -1:
                theta = adjacency_angles[q]

                if 0.5236 < theta < 2.6179:  
                    force[i] += 1 * adjacency_force_i[q] / ti.cos(0.4 * (pi - theta)) ** 2                    
                    force[j] += 1 * adjacency_force_j[q] / ti.cos(0.4 * (pi - theta)) ** 2 
                    
                    
                x_ij = x[i] - x[j]
                x_ij[2] = 0
                v_ij = v[i] - v[j]
                original_dist = neighbor_hold_l[i][k] * 1
                d = x_ij.normalized()
                current_dist = x_ij.norm()

                # Spring force
                force[i] += -0.5*spring_Y * d * (current_dist / original_dist - 1) 
                force[j] += 0.5*spring_Y * d * (current_dist / original_dist - 1) 
                # # # Dashpot damping
                force[i] += -v_ij.dot(d) * d * dashpot_damping * drag_damping   
                force[j] += v_ij.dot(d) * d * dashpot_damping * drag_damping 
                

        for k in ti.static(range(max_cols)):            
            j = neighbor[i][k]
            if j != -1:
                x_ij = x[i] - x[j]
                
                x_ij[2] = 0
                v_ij = v[i] - v[j]
                ori_ij = xin[i] - xin[j]
                d = x_ij.normalized()
                                                
                original_dist = ori_ij.norm() * 1
                current_dist = x_ij.norm()

                # Spring force
                force[i] += -0.5*spring_Y * d * (current_dist / original_dist - 1) 
                force[j] += 0.5*spring_Y * d * (current_dist / original_dist - 1) 
                # # # Dashpot damping
                force[i] += -v_ij.dot(d) * d * dashpot_damping * drag_damping   
                force[j] += v_ij.dot(d) * d * dashpot_damping * drag_damping 
                
                rho_i += current_dist
                
        rho[i] = rho_i


    for i  in ti.ndrange(n_vertexs):                
        v[i] += force[i] * dt * rho_ini[i] / rho[i]      
        x[i] += dt * v[i]
        force[i] = ti.Vector([0.0, 0.0, 0.0])     


@ti.kernel
def substep4():      
          

    for i in ti.ndrange(n_vertexs):
        force_delta = ti.Vector([0.0, 0.0, 0.0]) 
        delta = 0.0
        rho_i = 0.0
        
        if x[i][2] < floor:  # Bottom and left
            x[i][2] = floor  # move particle inside
            delta += abs(floor - x[i][2])
            v[i][2] = 0  # stop it from moving further
            delta_id[i] = 1
        
        
        if x[i][2] > xp[0][2]:  # Bottom and left
            x[i][2] = xp[0][2]  # move particle inside
            delta += abs(xp[0][2] - x[i][2])
            v[i][2] = 0  # stop it from moving further
            delta_id[i] = 1
            
               
        for k in ti.static(range(max_cols)):  
            judge_v = ti.Vector([0.5, 0.5]) 
            j = neighbor[i][k]
            if j != -1:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                ori_ij = xin[i] - xin[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = ori_ij.norm() * 1

                judge = delta_id[i] - delta_id[j]
                judge_v = judge * judge_v
                
                # Spring force
                force[i] += (-0.5)*spring_Y * d * (current_dist / original_dist - 1) 
                force[j] += (0.5)*spring_Y * d * (current_dist / original_dist - 1) 
                # # # Dashpot damping
                force[i] += -v_ij.dot(d) * d * dashpot_damping * drag_damping   
                force[j] += v_ij.dot(d) * d * dashpot_damping * drag_damping 
                
                force_delta += d                
                rho_i += current_dist                
        rho[i] = rho_i

        ##########################################################################
        for k in ti.static(range(max_cols_hold)):
            judge_v = ti.Vector([0.5, 0.5])             
            j = neighbor_hold[i][k]
            if j != -1:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                original_dist = neighbor_hold_l[i][k] * 1
                d = x_ij.normalized()
                current_dist = x_ij.norm()

                judge = delta_id[i] - delta_id[j]
                judge_v = judge * judge_v
                
                # Spring force
                force[i] += (-0.5)*spring_Y * d * (current_dist / original_dist - 1) 
                force[j] += (0.5)*spring_Y * d * (current_dist / original_dist - 1) 
                # # # Dashpot damping
                force[i] += -v_ij.dot(d) * d * dashpot_damping * drag_damping   
                force[j] += v_ij.dot(d) * d * dashpot_damping * drag_damping 

        ##########################################################################


    for i  in ti.ndrange(n_vertexs):                
        v[i] += force[i] * dt * rho_ini[i] / rho[i]   
        x[i] += dt * v[i]
        force[i] = ti.Vector([0.0, 0.0, 0.0]) 

    f = ti.Vector([0, 0, -10000.0])
    vp[0] += dt * f
    xp[0] += vp[0] * dt
    if xp[0][2] < floor:  # Bottom and left
        xp[0][2] = floor # move particle inside
        vp[0][2] = 0  # stop it from moving further 

@ti.kernel
def update_vertices():
    for i in ti.ndrange(n_vertexs):
        xfin[i] = x[i]

def save(pcdi, pcdfi, index_nan, path):

    o3d.io.write_point_cloud(path+'/interpcd_i.ply',pcdi,write_ascii= True)
    o3d.io.write_point_cloud(path+'/interpcd_f.ply',pcdfi,write_ascii= True)
    
    # o3d.visualization.draw_geometries([pcdfi],mesh_show_wireframe=True,mesh_show_back_face=True)

    ind = pd.DataFrame(data=index_nan, index=None, columns=None)
    ind.to_csv(path+'/ind_nan.csv')

def flattening(pcdnm, **kwargs):
    print('flattening begin')
    
    path = kwargs['work_path']   
    
    load_mesh(pcdnm)
    
    current_t = 0.0
    initialize_mass_points()
    rhoini()

    for i in trange(substeps*100):#########fps
        substep()
        current_t += dt
        
    for i in trange(substeps*40):#########fps       
        substep4()
        current_t += dt     
        
    update_vertices()

    xini = xin.to_numpy()*(1 / scale_l)
    xfina = xfin.to_numpy()*(1 / scale_l)
    index_nan = np.argwhere(np.isnan(xfina[:,0]))
    
    vin=xini.reshape(-1,3)
    xyzi=copy.deepcopy(vin)
    vfin=xfina.reshape(-1,3)
    xyzf=copy.deepcopy(vfin)
    color=copy.deepcopy(colors[:,:3])

    pcdi=o3d.geometry.PointCloud()
    pcdi.points=o3d.utility.Vector3dVector(xyzi)
    pcdi.colors=o3d.utility.Vector3dVector(color/255)
    
    pcdfi=o3d.geometry.PointCloud()
    pcdfi.points=o3d.utility.Vector3dVector(xyzf)
    pcdfi.colors=o3d.utility.Vector3dVector(color/255)

    save(pcdi, pcdfi, index_nan, path)

    print('flattening done')

    return pcdfi, index_nan


if __name__ == '__main__':
    time0 = time.time()
    current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
    kwargs = {"input_pcd": 'E:/paper_flatten/integrated_code/temp/interpcd_downsample.ply', 
            "work_path": current_path + '/temp', 
            "output_path": current_path + '/result'}

    pcdnm=o3d.io.read_point_cloud(kwargs['input_pcd'])

    flattening(pcdnm, **kwargs)
    
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print(scale_l)
    print(time.time()-time0)