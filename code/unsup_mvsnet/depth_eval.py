from __future__ import print_function
from preprocess import load_pfm
import numpy as np
import os


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__ == '__main__':
    exp_name = 'exp1'
    iter_no = '50000'
    pred_dir = os.path.join('/home/slin/Documents/outputs/mvs', exp_name, 'tests', iter_no)
    gt_dir = '/home/slin/Documents/datasets/dtu/training/Depths/'

    enable_median_scaling = False
    enable_prob_filter = True
    prob_threshold = 0.8

    errors = []
    if enable_median_scaling:
        ratios = []

    for scan in os.listdir(pred_dir):
        if scan!='scan10':
            continue
        pred_depth_dir = os.path.join(pred_dir, scan, 'depths_mvsnet')
        gt_depth_dir = os.path.join(gt_dir, scan + '_train')
        for pred_depth_filename in os.listdir(pred_depth_dir):
            if not pred_depth_filename.endswith('_init.pfm'):
                continue
            pred_depth = load_pfm(open(os.path.join(pred_depth_dir, pred_depth_filename)))
            if enable_prob_filter:
                pred_prob_filename = pred_depth_filename[:-8] + 'prob.pfm'
                pred_prob = load_pfm(open(os.path.join(pred_depth_dir, pred_prob_filename)))
                pred_depth[pred_prob < prob_threshold] = 0
            gt_depth_filename = 'depth_map_' + pred_depth_filename[4:8] + '.pfm'
            gt_depth = load_pfm(open(os.path.join(gt_depth_dir, gt_depth_filename)))

            mask = gt_depth > 0
            pred_depth = pred_depth[np.logical_and(mask, pred_prob>0.8)]
            gt_depth = gt_depth[np.logical_and(mask, pred_prob>0.8)]
            if enable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio
            errors.append(compute_errors(gt_depth, pred_depth))

    if enable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
