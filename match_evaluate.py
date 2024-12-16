"""
This script performs image matching using a specified matcher model. It processes pairs of input images,
detects keypoints, matches them, and performs RANSAC to find inliers. The results, including visualizations
and metadata, are saved to the specified output directory.
"""

import sys
import torch
import argparse
import matplotlib
from pathlib import Path
import numpy as np
import os
import numpy as np
import cv2
import csv
from glob import glob
import matplotlib.pyplot as plt
from collections import namedtuple
from copy import deepcopy
from tqdm import tqdm
import random
import torch
from matching.utils import get_image_pairs_paths
from matching import get_matcher, available_models
from matching.viz import plot_matches

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")


def main(args):
    image_size = [args.im_size, args.im_size]
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts)

    pairs_of_paths = get_image_pairs_paths(args.input)
    for i, (img0_path, img1_path) in enumerate(pairs_of_paths):

        image0 = matcher.load_image(img0_path, resize=image_size)
        image1 = matcher.load_image(img1_path, resize=image_size)
        result = matcher(image0, image1)

        out_str = f"Paths: {str(img0_path), str(img1_path)}. Found {result['num_inliers']} inliers after RANSAC. "

        if not args.no_viz:
            viz_path = args.out_dir / f"output_{i}_matches.jpg"
            plot_matches(image0, image1, result, save_path=viz_path)
            out_str += f"Viz saved in {viz_path}. "

        result["img0_path"] = img0_path
        result["img1_path"] = img1_path
        result["matcher"] = args.matcher
        result["n_kpts"] = args.n_kpts
        result["im_size"] = args.im_size

        dict_path = args.out_dir / f"output_{i}_result.torch"
        torch.save(result, dict_path)
        out_str += f"Output saved in {dict_path}"

        print(out_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Matching Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Choose matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )

    # Hyperparameters shared by all methods:
    parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")

    parser.add_argument(
        "--input",
        type=str,
        default="assets/example_pairs",
        help="path to either (1) dir with dirs with image pairs or (2) txt file with two image paths per line",
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.matcher}")

    return args

# Load ground truth

# Named tuple to store camera intrinsics and extrinsics
CameraCalibration = namedtuple('CameraCalibration', ['K', 'R', 'T'])

def load_calibration(filepath):
    """Load calibration data from a CSV file."""
    calibration_data = {}
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            camera_id = row[0]
            K = np.array(list(map(float, row[1].split()))).reshape(3, 3)
            R = np.array(list(map(float, row[2].split()))).reshape(3, 3)
            T = np.array(list(map(float, row[3].split())))
            calibration_data[camera_id] = CameraCalibration(K=K, R=R, T=T)
    return calibration_data

def load_scaling_factors(filepath):
    """Load scaling factors from a CSV file."""
    scaling_factors = {}
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            scaling_factors[row[0]] = float(row[1])
    return scaling_factors

# Load calibration data and scaling factors
calib_dict = load_calibration(f'{src}/{scene}/calibration.csv')
print(f'Loaded ground truth data for {len(calib_dict)} images.\n')

scaling_dict = load_scaling_factors(f'{src}/scaling_factors.csv')
print(f'Scaling factors: {scaling_dict}\n')

def normalize_keypoints(keypoints, K):
    """Normalize keypoints using the camera intrinsics."""
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return (keypoints - np.array([[cx, cy]])) / np.array([[fx, fy]])

def compute_essential_matrix(F, K1, K2, kp1, kp2):
    """Compute the Essential matrix from the Fundamental matrix and keypoints."""
    E = K2.T @ F @ K1
    kp1_normalized = normalize_keypoints(kp1, K1)
    kp2_normalized = normalize_keypoints(kp2, K2)
    _, R, T, _ = cv2.recoverPose(E, kp1_normalized, kp2_normalized)
    return E, R, T


def quaternion_from_matrix(matrix):
    """Convert a rotation matrix into a quaternion."""
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    trace = np.trace(M)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (M[2, 1] - M[1, 2]) / s
        qy = (M[0, 2] - M[2, 0]) / s
        qz = (M[1, 0] - M[0, 1]) / s
    else:
        idx = np.argmax(np.diagonal(M))
        if idx == 0:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            qw = (M[2, 1] - M[1, 2]) / s
            qx = 0.25 * s
            qy = (M[0, 1] + M[1, 0]) / s
            qz = (M[0, 2] + M[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            qw = (M[0, 2] - M[2, 0]) / s
            qx = (M[0, 1] + M[1, 0]) / s
            qy = 0.25 * s
            qz = (M[1, 2] + M[2, 1]) / s
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            qw = (M[1, 0] - M[0, 1]) / s
            qx = (M[0, 2] + M[2, 0]) / s
            qy = (M[1, 2] + M[2, 1]) / s
            qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def compute_error(q_gt, T_gt, q, T, scale):
    """Compute the error metrics for rotation and translation."""
    eps = 1e-15
    q_gt_normalized = q_gt / (np.linalg.norm(q_gt) + eps)
    q_normalized = q / (np.linalg.norm(q) + eps)
    loss_q = max(eps, 1.0 - np.sum(q_gt_normalized * q_normalized)**2)
    err_q = np.arccos(1 - 2 * loss_q)

    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)
    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return np.degrees(err_q), err_t

# OpenCV gives us the Fundamental matrix after RANSAC, and a mask over the input matches. The solution is clearly much cleaner, even though it may still contain outliers.
def evaluate_one_pair_from_torch(matched_kpts1_coords, matched_kpts2_coords, pair_id):
    image_id1, image_id2 = pair_id.split('-')

    F, inlier_mask = cv2.findFundamentalMat(
        matched_kpts1_coords, 
        matched_kpts2_coords, 
        method=cv2.USAC_MAGSAC, 
        ransacReprojThreshold=0.25, 
        confidence=0.99999, 
        maxIters=10000)

    inlier_kp_1 = matched_kpts1_coords[inlier_mask.ravel() == 1]
    inlier_kp_2 = matched_kpts1_coords[inlier_mask.ravel() == 1]

    # Compute the Essential matrix, rotation, and translation
    E, R, T = compute_essential_matrix(F, calib_dict[image_id1].K, calib_dict[image_id2].K, inlier_kp_1, inlier_kp_2)
    q = quaternion_from_matrix(R)
    T = T.flatten()

    # Compute ground truth pose differences
    R1_gt, T1_gt = calib_dict[image_id1].R, calib_dict[image_id1].T.reshape((3, 1))
    R2_gt, T2_gt = calib_dict[image_id2].R, calib_dict[image_id2].T.reshape((3, 1))
    dR_gt = R2_gt @ R1_gt.T
    dT_gt = (T2_gt - dR_gt @ T1_gt).flatten()
    q_gt = quaternion_from_matrix(dR_gt)

        # Compute errors
    err_q, err_t = compute_error(q_gt, dT_gt, q, T, scaling_dict[scene])
    return err_q, err_t

def load_torch_file(file_path):
    try:
        loaded_torch = torch.load(file_path)
        
        kpts1 = loaded_torch['all_kpts0']
        kpts2 = loaded_torch['all_kpts1']
        matched_kpts1_coords = loaded_torch['matched_kpts0']
        matched_kpts2_coords = loaded_torch['matched_kpts1']
        inlier_kpts1 = loaded_torch['inlier_kpts0']
        inlier_kpts2 = loaded_torch['inlier_kpts1']
        print(f"Successfully loaded {file_path}")
        return kpts1, kpts2, matched_kpts1_coords, matched_kpts2_coords, inlier_kpts1, inlier_kpts2
    except Exception as e:
        print(f"Error loading .torch file: {e}")
        return None

#TODO change to pair ID
pair_id = pairs[0] # 90920828_5082887495-20133057_3035445116
file_path = 'output_1_result.torch'  # Replace with your .torch file path
kpts1, kpts2, matched_kpts1_coords, matched_kpts2_coords, inlier_kpts1, inlier_kpts2 = load_torch_file(file_path)
# print("kpts1", kpts1.shape, kpts1[0])
# print("matched_kpts1",matched_kpts1_coords.shape, matched_kpts1_coords[0])
# print("matched_kpts2",matched_kpts2_coords.shape, matched_kpts2_coords[0])
# print("inlier_kpts1",inlier_kpts1.shape, inlier_kpts1[0])

err_q, err_t = evaluate_one_pair_from_torch(matched_kpts1_coords, matched_kpts2_coords, pair_id)
print(f'Pair "{pair_id}", rotation error: {err_q:.2f}° | translation error: {err_t:.2f} m')


if __name__ == "__main__":
    args = parse_args()
    main(args)
