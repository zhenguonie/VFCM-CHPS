import os
import cv2

from source1_0403 import preprocess
from column_spring_mass import flattening
from source2_ms_0707 import postprocess
from evaluation import evalSingle

def flatten(**kwargs):
    pcd_ds, pcd_ori, point_map = preprocess(**kwargs)
    pcdfi, ind_nan = flattening(pcd_ds, **kwargs)
    pcdo, dst, img = postprocess(pcd_ori, pcd_ds, pcdfi, point_map, ind_nan, **kwargs)

    return pcdo, dst, img

if __name__ == '__main__':
    current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
    kwargs = {"original_image": current_path + '/img/case_001.jpg', 
            "input_pcd": current_path + '/ply/case_001.ply', 
            "work_path": current_path + '/temp', 
            "output_path": current_path + '/result'}

    pcdo, dst, img = flatten(**kwargs)
    oriImg = cv2.imread(kwargs['original_image'])
    evalSingle(dst, oriImg)
