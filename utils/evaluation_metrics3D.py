import numpy as np
import SimpleITK as sitk
import glob
import os
from scipy.spatial import distance
from sklearn.metrics import f1_score


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 255) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 255)))
    TP = np.float(np.sum((pred == 255) & (gt == 255)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def Dice(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))
    return dice


def IoU(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    m1 = np.sum(pred[gt == 1])
    m2 = np.sum(pred == 1) + np.sum(gt == 1) - m1
    iou = m1 / m2
    return iou


def metrics_3d(pred, gt):
    FP, FN, TP, TN = numeric_score(pred, gt)
    recall = TP / (TP + FN + 1e-10)  # Recall
    precision=TP / (TP + FP + 1e-10)
    fnr = FN / (FN + TP + 1e-10)
    fpr = FN / (FP + TN + 1e-10)
    iou = TP / (TP + FN + FP + 1e-10)
    dice = 2*TP / (2*TP + FN + FP + 1e-10)
    return recall, precision, fpr, dice
    # return tpr, fnr, fpr, iou


def over_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    OR = Os / (Rs + Os)
    return OR


def under_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Us = np.float(np.sum((pred == 0) & (gt == 255)))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    UR = Us / (Rs + Os)
    return UR
