import scipy.io as sio
import numpy as np
from util import load_image, compute_fundamental
from matching import find_matches
import scipy
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
def int_to_str(n):
    ret = str(n)
    return "0" * (3-len(ret)) + ret

def matches(img0, img1):
  kp0, des0 = sift.detectAndCompute(img0, None)
  kp1, des1 = sift.detectAndCompute(img1, None)
  matches = flann.knnMatch(des0,des1,k=2)
  good = [] 
  for m,n in matches: 
    if m.distance < 0.75*n.distance: 
      good.append(m)
  h1, w1 = img0.shape[:2] 
  h2, w2 = img1.shape[:2] 
  view = scipy.zeros((max(h1, h2), w1 + w2, 3), scipy.uint8) 
  view[:h1, :w1, 0] = img0
  view[:h2, w1:, 0] = img1
  view[:, :, 1] = view[:, :, 0] 
  view[:, :, 2] = view[:, :, 0] 
  pts1 = []
  pts2 = []
  for m in good: 
    pts1.append([int(kp0[m.queryIdx].pt[0]), int(kp0[m.queryIdx].pt[1]), 0])
    pts2.append([int(kp1[m.trainIdx].pt[0] + w1), int(kp1[m.trainIdx].pt[1]), 0]) 
  return pts1, pts2

sift = cv2.xfeatures2d.SIFT_create()
index_params = dict(algorithm = 1, trees = 5) 
search_params = dict(checks=50)  # or pass empty dictionary 
flann = cv2.FlannBasedMatcher(index_params,search_params)
N = 10 # dataset size
DATA_DIR = "./data/house/images" #data size 10
CAMERA_DIR = "./data/house/3D" 
IMAGE_DIRS = [os.path.join(DATA_DIR, "house." + int_to_str(n) + ".pgm" )for n in range(N)]
images = [cv2.imread(img, 0) for img in IMAGE_DIRS]
CAMERA_DIRS = [os.path.join(CAMERA_DIR, "house." + int_to_str(n) + ".P") for n in range(N)]
camera_matrices = [np.loadtxt(cam) for cam in CAMERA_DIRS]
ItoWs = [np.vstack((m, [[0,0,0,1]])) for m in camera_matrices]
WtoIs = [np.linalg.inv(m) for m in ItoWs]

transformation_matrices = [np.eye(3,3)]
matched_points = [] # in its own coordinate
for i in range(1, N):
    prev = transformation_matrices[i-1]
    img0 = images[i-1]
    img1 = images[i]
    pts1, pts2 = matches(img0, img1)
    n = len(pts1)
    pts1 = [np.matmul(ItoWs[i-1], np.concatenate((pts1[k],[1])))[:3] for k in range(n)]
    pts2 = [np.matmul(ItoWs[i], np.concatenate((pts2[k],[1])))[:3] for k in range(n)]
    matched_points.append(pts1)
    if i == N-1:
        matched_points.append(pts2)
    F = compute_fundamental(pts2, pts1) # maps points from 2 to 1
    transformation_matrices.append(np.matmul(prev, F))

camera_centers = []
points = [] # in image 0's coordinate
for i in range(N):
    m = camera_matrices[i]
    WtoI = WtoIs[i]
    R = m[:, :3]
    t = m[:,3:]
    c = -np.matmul(np.linalg.inv(R), t).T[0] # camera center
    cc = np.matmul(transformation_matrices[i], c)
    camera_centers.append(cc.reshape((1,3)))
    pts = [np.matmul(transformation_matrices[i], pt).reshape((1,3)) for pt in matched_points[i]]
    points.append(pts)


res_points = []
for i in range(N):
    for j in range(i, N):
        pts0 = points[i]
        pts1 = points[j]
        cc0 = camera_centers[i]
        cc1 = camera_centers[j]
        eye = np.eye(3)
        n = min(len(pts0), len(pts1))
        print(n)
        for k in range(n):
            try:
                x0 = pts0[k]
                x1 = pts1[k]
            except:
                continue
            v0 = x0 - cc0
            v0 = v0/np.linalg.norm(v0)
            v1 = x1 - cc1
            v1 = v1/np.linalg.norm(v1)
            # calculate the intersection point, reference from Book, might be wrong
            tmp1 = np.linalg.inv(((eye - np.matmul(v0.T, v0)) + (eye - np.matmul(v1.T, v1))))
            tmp2 = np.matmul((eye - np.matmul(v0.T, v0)), cc0.T) + np.matmul((eye - np.matmul(v1.T, v1)), cc1.T)
            point = np.matmul(tmp1, tmp2)
            res_points.append(point)

print(res_points)
#res_points = [np.matmul(WtoIs[0], np.concatenate((p, np.array([1]).reshape((1,1)))))[:3] for p in res_points]
xs = [p[0] for p in res_points]
ys = [p[1] for p in res_points]
zs = [p[2] for p in res_points]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs, ys, zs, zdir='z', s=20)
plt.show()





