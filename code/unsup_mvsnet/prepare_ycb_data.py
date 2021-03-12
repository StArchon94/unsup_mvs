import math
import os
import shutil
import sys
import time

import cv2
import h5py
import numpy as np

from preprocess import write_cam, write_pfm


def im2col(im, psize):
    n_channels = 1 if len(im.shape) == 2 else im.shape[0]
    (n_channels, rows, cols) = (1,) * (3 - len(im.shape)) + im.shape

    im_pad = np.zeros((n_channels,
                       int(math.ceil(1.0 * rows / psize) * psize),
                       int(math.ceil(1.0 * cols / psize) * psize)))
    im_pad[:, 0:rows, 0:cols] = im

    final = np.zeros((im_pad.shape[1], im_pad.shape[2], n_channels,
                      psize, psize))
    for c in xrange(n_channels):
        for x in xrange(psize):
            for y in xrange(psize):
                im_shift = np.vstack(
                    (im_pad[c, x:], im_pad[c, :x]))
                im_shift = np.column_stack(
                    (im_shift[:, y:], im_shift[:, :y]))
                final[x::psize, y::psize, c] = np.swapaxes(
                    im_shift.reshape(im_pad.shape[1] / psize, psize,
                                     im_pad.shape[2] / psize, psize), 1, 2)

    return np.squeeze(final[0:rows - psize + 1, 0:cols - psize + 1])


def filterDiscontinuities(depthMap):
    filt_size = 7
    thresh = 1000

    # Ensure that filter sizes are okay
    assert filt_size % 2 == 1, "Can only use odd filter sizes."

    # Compute discontinuities
    offset = (filt_size - 1) / 2
    patches = 1.0 * im2col(depthMap, filt_size)
    mids = patches[:, :, offset, offset]
    mins = np.min(patches, axis=(2, 3))
    maxes = np.max(patches, axis=(2, 3))

    discont = np.maximum(np.abs(mins - mids),
                         np.abs(maxes - mids))
    mark = discont > thresh

    # Account for offsets
    final_mark = np.zeros((480, 640), dtype=np.uint16)
    final_mark[offset:offset + mark.shape[0],
               offset:offset + mark.shape[1]] = mark

    return depthMap * (1 - final_mark)


def getRGBFromDepthTransform(calibration, camera):
    irKey = "H_NP{}_ir_from_NP5".format(camera)
    rgbKey = "H_NP{}_from_NP5".format(camera)

    rgbFromRef = calibration[rgbKey][:]
    irFromRef = calibration[irKey][:]

    return np.dot(rgbFromRef, np.linalg.inv(irFromRef))


def registerDepthMap(unregisteredDepthMap, rgb_shape, depthK, rgbK, H_RGBFromDepth):
    unregisteredHeight, unregisteredWidth = unregisteredDepthMap.shape[:2]
    registeredHeight, registeredWidth = rgb_shape[:2]

    registeredDepthMap = np.zeros((registeredHeight, registeredWidth))
    xyzDepth = np.empty((4, 1))
    xyzRGB = np.empty((4, 1))

    # Ensure that the last value is 1 (homogeneous coordinates)
    xyzDepth[3] = 1

    invDepthFx = 1.0 / depthK[0, 0]
    invDepthFy = 1.0 / depthK[1, 1]
    depthCx = depthK[0, 2]
    depthCy = depthK[1, 2]

    rgbFx = rgbK[0, 0]
    rgbFy = rgbK[1, 1]
    rgbCx = rgbK[0, 2]
    rgbCy = rgbK[1, 2]

    undistorted = np.empty(2)
    for v in range(unregisteredHeight):
        for u in range(unregisteredWidth):

            depth = unregisteredDepthMap[v, u]
            if depth == 0:
                continue

            xyzDepth[0] = ((u - depthCx) * depth) * invDepthFx
            xyzDepth[1] = ((v - depthCy) * depth) * invDepthFy
            xyzDepth[2] = depth

            xyzRGB[0] = (H_RGBFromDepth[0, 0] * xyzDepth[0] +
                         H_RGBFromDepth[0, 1] * xyzDepth[1] +
                         H_RGBFromDepth[0, 2] * xyzDepth[2] +
                         H_RGBFromDepth[0, 3])
            xyzRGB[1] = (H_RGBFromDepth[1, 0] * xyzDepth[0] +
                         H_RGBFromDepth[1, 1] * xyzDepth[1] +
                         H_RGBFromDepth[1, 2] * xyzDepth[2] +
                         H_RGBFromDepth[1, 3])
            xyzRGB[2] = (H_RGBFromDepth[2, 0] * xyzDepth[0] +
                         H_RGBFromDepth[2, 1] * xyzDepth[1] +
                         H_RGBFromDepth[2, 2] * xyzDepth[2] +
                         H_RGBFromDepth[2, 3])

            invRGB_Z = 1.0 / xyzRGB[2]
            undistorted[0] = (rgbFx * xyzRGB[0]) * invRGB_Z + rgbCx
            undistorted[1] = (rgbFy * xyzRGB[1]) * invRGB_Z + rgbCy

            uRGB = int(undistorted[0] + 0.5)
            vRGB = int(undistorted[1] + 0.5)

            if (uRGB < 0 or uRGB >= registeredWidth) or (vRGB < 0 or vRGB >= registeredHeight):
                continue

            registeredDepth = xyzRGB[2]
            if registeredDepth > 0 and (not registeredDepthMap[vRGB, uRGB] or registeredDepth < registeredDepthMap[vRGB, uRGB]):
                registeredDepthMap[vRGB, uRGB] = registeredDepth

    return registeredDepthMap


if __name__ == '__main__':
    os.chdir('/home/slin/Documents/datasets/ycb/test/001_chips_can/')
    if not os.path.isdir('cams'):
        os.mkdir('cams')
    if not os.path.isdir('Depths'):
        os.mkdir('Depths')
    if not os.path.isdir('images'):
        os.mkdir('images')

    hf_calibration = h5py.File('calibration.h5', 'r')
    count = 0
    last_time = time.time()
    for np_id in range(1, 6):
        H_relative = hf_calibration['H_NP{}_from_NP5'.format(np_id)][:]
        rgbK = hf_calibration['NP{}_rgb_K'.format(np_id)][:]
        depthK = hf_calibration['NP{}_ir_K'.format(np_id)][:]
        for degree in range(0, 360, 3):
            image_path = 'NP{}_{}.jpg'.format(np_id, degree)
            image_save_path = 'images/{:08d}.jpg'.format(count)
            if not os.path.exists(image_save_path):
                shutil.copy(image_path, image_save_path)

            pose_path = 'poses/NP5_{}_pose.h5'.format(degree)
            H_table_from_reference_camera = h5py.File(pose_path, 'r')['H_table_from_reference_camera'][:]

            extrinsics = np.matmul(H_relative, H_table_from_reference_camera)
            intrinsics = np.zeros((4, 4))
            intrinsics[:3, :3] = rgbK

            raw_depth = h5py.File('NP{}_{}.h5'.format(np_id, degree), 'r')['depth'][:]
            depthScale = hf_calibration['NP{}_ir_depth_scale'.format(np_id)][:] * .0001  # 100um to meters
            raw_depth = filterDiscontinuities(raw_depth) * depthScale
            H_RGBFromDepth = getRGBFromDepthTransform(hf_calibration, np_id)

            depth = registerDepthMap(raw_depth, cv2.imread(image_path).shape, depthK, rgbK, H_RGBFromDepth).astype(np.float32)
            depth_save_path = 'Depths/depth_map_{:04d}.pfm'.format(count)
            write_pfm(depth_save_path, depth)

            min_depth = np.min(depth[depth > 0]) * 0.9
            max_depth = np.max(depth) * 1.1
            step_size_depth = (max_depth - min_depth) / 255
            intrinsics[3, 0] = min_depth
            intrinsics[3, 1] = step_size_depth

            cam = [extrinsics, intrinsics]
            cam_file_path = 'cams/{:08d}_cam.txt'.format(count)
            write_cam(cam_file_path, cam)

            count += 1
            cur_time = time.time()
            print('NP{}_{}, {:.2f}s'.format(np_id, degree, cur_time - last_time))
            last_time = cur_time

    f = open('pair.txt', 'w')

    num_cameras = 5
    num_degrees = 360
    step_size_degree = 3
    num_images = int(num_cameras * num_degrees / step_size_degree)
    f.write('{}\n'.format(num_images))  # number of total images in a YCB scan

    count = 0
    for i in range(num_images):
        f.write('{}\n'.format(i))
        f.write('10')
        f.write(' ')

        idx_mod_120 = i % 120
        view_1 = (idx_mod_120 - 1) % 120 + i  # one to the left of the ref
        view_2 = (idx_mod_120 - 2) % 120 + i  # two to the left of the ref
        view_3 = (idx_mod_120 + 1) % 120 + i  # one to the right of the ref
        view_4 = (idx_mod_120 + 2) % 120 + i  # two to the right of the ref

        view_5 = (idx_mod_120 - 1) % 120 + (i - 120) % 600  # top left
        view_6 = (idx_mod_120 - 0) % 120 + (i - 120) % 600  # top
        view_7 = (idx_mod_120 + 1) % 120 + (i - 120) % 600  # top right

        view_8 = (idx_mod_120 - 1) % 120 + (i + 120) % 600  # bottom left
        view_9 = (idx_mod_120 - 0) % 120 + (i + 120) % 600  # bottom
        view_10 = (idx_mod_120 + 1) % 120 + (i + 120) % 600  # bottom right

        candidate_view = [view_1, view_2, view_3, view_4, view_5, view_6, view_7, view_8, view_9, view_10]
        for view in candidate_view:
            f.write('{} 0 '.format(view))

        f.write('\n')

    f.close()
