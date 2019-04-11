import numpy as np
from collections import Counter
import os
from PIL import Image
import itertools
#import png
import torch.nn.functional as F

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = F.upsample(img, size=(nh, nw), mode='nearest')
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs

def compute_errors(ground_truth, predication):

    # accuracy
    threshold = np.maximum((ground_truth / predication),(predication / ground_truth))
    a1 = (threshold < 1.25 ).mean()
    a2 = (threshold < 1.25 ** 2 ).mean()
    a3 = (threshold < 1.25 ** 3 ).mean()

    #MSE
    rmse = (ground_truth - predication) ** 2
    rmse = np.sqrt(rmse.mean())

    #MSE(log)
    rmse_log = (np.log(ground_truth) - np.log(predication)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Abs Relative difference
    abs_rel = np.mean(np.abs(ground_truth - predication) / ground_truth)

    # Squared Relative difference
    sq_rel = np.mean(((ground_truth - predication) ** 2) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

class KITTI:

    def read_calib_file(self, path):
        # taken from https://github.com/hunse/kitti
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass

        return data

    def get_fb(self, calib_dir, cam=2):
        cam2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        P2_rect = cam2cam['P_rect_02'].reshape(3,4)
        P3_rect = cam2cam['P_rect_03'].reshape(3,4)

        # cam 2 is left of camera 0  -6cm
        # cam 3 is to the right  +54cm
        b2 = P2_rect[0,3] / -P2_rect[0,0]
        b3 = P3_rect[0,3] / -P3_rect[0,0]
        baseline = b3-b2

        if cam==2:
            focal_length = P2_rect[0,0]
        elif cam==3:
            focal_length = P3_rect[0,0]

        return focal_length * baseline

    def load_velodyne_points(self, file_name):
        # adapted from https://github.com/hunse/kitti
        points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # homogeneous
        return points

    def lin_interp(self, shape, xyd):
        # taken from https://github.com/hunse/kitti
        from scipy.interpolate import LinearNDInterpolator
        m, n = shape
        ij, d = xyd[:, 1::-1], xyd[:, 2]
        f = LinearNDInterpolator(ij, d, fill_value=0)
        J, I = np.meshgrid(np.arange(n), np.arange(m))
        IJ = np.vstack([I.flatten(), J.flatten()]).T
        disparity = f(IJ).reshape(shape)
        return disparity

    def sub2ind(self, matrixSize, rowSub, colSub):
        m, n = matrixSize
        return rowSub * (n-1) + colSub - 1

    def get_depth(self, calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False):
        # load calibration files
        cam2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
        P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

        # load velodyne points and remove all behind image plane (approximation)
        # each row of the velodyne data is forward, left, up, reflectance
        velo = self.load_velodyne_points(velo_file_name)
        velo = velo[velo[:, 0] >= 0, :]

        # project the points to the camera
        velo_pts_im = np.dot(P_velo2im, velo.T).T
        velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

        if vel_depth:
            velo_pts_im[:, 2] = velo[:, 0]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        velo_pts_im[:, 0] = np.round(velo_pts_im[:,0])
        velo_pts_im[:, 1] = np.round(velo_pts_im[:,1])
        velo_pts_im -= 1
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros((im_shape))
        depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = self.sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds==dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth<0] = 0

        if interp:
            # interpolate the depth map to fill in holes
            depth_interp = lin_interp(im_shape, velo_pts_im)
            return depth, depth_interp
        else:
            return depth
