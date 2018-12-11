import scipy.io as sio
import numpy as np
from util import load_image, compute_fundamental
from matching import find_matches
import scipy
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
import os

def int_to_str(n):
    ret = str(n)
    return "0" * (3-len(ret)) + ret

N = 2 # dataset size
DATA_DIR = "./data/house/images" #data size 10
CAMERA_DIR = "./data/house/3D" 
IMAGE_DIRS = [os.path.join(DATA_DIR, "house." + int_to_str(n) + ".pgm" )for n in range(N)]
images = [load_image(img) for img in IMAGE_DIRS]
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
    pts1, pts2 = find_matches(img0, img1)
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
xs = [p[0] for p in res_points]
ys = [p[1] for p in res_points]
zs = [p[2] for p in res_points]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs, ys, zs, zdir='z', s=20)
plt.show()





