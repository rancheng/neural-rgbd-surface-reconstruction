import numpy as np
import re
import cv2
import transformations as tfs

def load_poses_tum(posefile):
    pose_items = np.genfromtxt(posefile)
    pose_list = []
    for i, pose in enumerate(pose_items):
        # np.roll(pose[3:], -1) uncomment this if the quaternion is saved as [w, x, y, z]
        timestamps = pose[0]
        pose = pose[1:]
        quaternion = np.roll(pose[3:], 1)  # tum format is xyzw, need to roll to wxyz
        position = pose[:3]
        tf_matrix = tfs.quaternion_matrix(quaternion)
        tf_matrix[0, 3] = position[0] + 1
        tf_matrix[1, 3] = position[1]
        tf_matrix[2, 3] = position[2]
        pose_list.append(tf_matrix)
    return pose_list


def load_poses(posefile):
    file = open(posefile, "r")
    lines = file.readlines()
    file.close()
    poses = []
    valid = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if 'nan' in lines[i]:
            valid.append(False)
            poses.append(np.eye(4, 4, dtype=np.float32).tolist())
        else:
            valid.append(True)
            pose_floats = [[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]]
            poses.append(pose_floats)

    return poses, valid


def load_focal_length(filepath):
    file = open(filepath, "r")
    return float(file.readline())


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]


def resize_images(images, H, W, interpolation=cv2.INTER_LINEAR):
    resized = np.zeros((images.shape[0], H, W, images.shape[3]), dtype=images.dtype)
    for i, img in enumerate(images):
        r = cv2.resize(img, (W, H), interpolation=interpolation)
        if images.shape[3] == 1:
            r = r[..., np.newaxis]
        resized[i] = r
    return resized
