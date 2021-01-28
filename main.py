#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import pprint

img_num = 201
img_dim = 3575
max_dist = 999

# 画像の読み込み、グレースケール、トリミング
X = np.zeros((img_num, img_dim), np.float32)
for i in range(img_num):
    filename = "LFW1200/LFW1" + str(i).zfill(3) + ".jpg"
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img[100:165, 95:150]
    img = img.reshape((1, img_dim))
    X[i, :] = img[0, :]

# 全画像データから平均を減算
mean = np.mean(X, axis=0)
X = X-mean

# 分散共分散行列の算出
n = X.shape[0]
Cov = np.dot(X, X.T)/n

# 固有値＆固有ベクトルの算出
eigval, eigvec_v = np.linalg.eig(Cov)

# 固有ベクトルの正規化
eigvec_u = np.dot(X.T, eigvec_v)
for i in range(img_num):
    eigvec_u[:, i] /= np.linalg.norm(eigvec_u[:, i])

revimgs = np.zeros((img_num, img_dim), np.float32)
dim_num = 80
target_img_num = 0

# 全画像データを80次元に
for i in range(img_num):
    eigvec = eigvec_u[:, :dim_num]
    PCAdim = np.dot(X[i, :], eigvec)
    revimg = np.dot(PCAdim, eigvec.T)
    revimg += mean
    revimgs[i, :] = revimg

dists = np.ones((img_num), np.float32) * max_dist
data_dict = {}

# 比較したい画像との距離を計算
for i in range(img_num):
    v = revimgs[target_img_num] - revimgs[i]
    dists[i] = np.linalg.norm(v, ord=2)
    data_dict[i] = dists[i]

sorted_data_dict = sorted(data_dict.items(), key=lambda x: x[1])
pprint.pprint(sorted_data_dict)

bush_img_nums = [0, 3, 13, 18, 26, 40, 42, 46, 50, 52, 59, 60, 70, 76, 79, 82, 84, 85, 90, 103, 107, 108,
                 113, 114, 116, 118, 121, 123, 124, 129, 136, 141, 147, 152, 154, 159, 164, 170, 171, 183, 191, 199]

for theta in range(850, 4000, 100):
    TP = FN = FP = TN = 0
    for index, dist in data_dict.items():
        if(dist < theta and index in bush_img_nums):
            TP += 1
        elif(dist > theta and index in bush_img_nums):
            FN += 1
        elif(dist < theta and index not in bush_img_nums):
            FP += 1
        else:
            TN += 1

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    print(precision, recall)
