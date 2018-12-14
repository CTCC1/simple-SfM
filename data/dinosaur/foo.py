import scipy.io as sio
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sift = cv2.xfeatures2d.SIFT_create()
index_params = dict(algorithm = 1, trees = 5) 
search_params = dict(checks=50)  # or pass empty dictionary 
flann = cv2.FlannBasedMatcher(index_params,search_params)

mat = sio.loadmat("dino_Ps.mat")["P"][0]
data = ["001", "002", "003", "004", "005", "006"]
images = [cv2.imread("./images/viff." + d + ".ppm", 0) for d in data]
camera_centers = []
interest_points = []

ItoWs = []
WtoIs = []

for d in data:
  m = mat[int(d)]
  ItoW = np.vstack((m, [[0,0,0,1]]))
  WtoI = ItoW.T
  ItoWs.append(ItoW)
  WtoIs.append(WtoI)
  R = m[:,:3]
  t = m[:,3:]
  c = -np.matmul(np.linalg.inv(R), t).T[0] # camera center
  Ic = np.matmul(WtoIs[0], np.concatenate((c,[1])))
  camera_centers.append(Ic[0:3])


res_points = []
for i in range(len(data) - 1):
  image0 = images[i]
  image1 = images[i+1]
  kp0, des0 = sift.detectAndCompute(image0, None)
  kp1, des1 = sift.detectAndCompute(image1, None)
  matches = flann.knnMatch(des0,des1,k=2)
  good = [] 
  for m,n in matches: 
    if m.distance < 0.75*n.distance: 
      good.append(m)
  h1, w1 = images[0].shape[:2] 
  h2, w2 = images[1].shape[:2] 
  view = scipy.zeros((max(h1, h2), w1 + w2, 3), scipy.uint8) 
  view[:h1, :w1, 0] = image0
  view[:h2, w1:, 0] = image1
  view[:, :, 1] = view[:, :, 0] 
  view[:, :, 2] = view[:, :, 0] 
  points0 = []
  points1 = []
  for m in good: 
    points0.append([int(kp0[m.queryIdx].pt[0]), int(kp0[m.queryIdx].pt[1]), 0, 1])
    points1.append([int(kp1[m.trainIdx].pt[0] + w1), int(kp1[m.trainIdx].pt[1]), 0, 1]) 
  cc0 = camera_centers[i]
  cc1 = camera_centers[i+1]
  num_points = len(points0)
  for idx in range(num_points):
    eye = np.eye(3)
    v0 = cc0 - np.matmul(WtoIs[0], np.matmul(ItoWs[i], np.asarray(points0[idx])).T)[0:3]
    v0 = np.matmul(ItoWs[i], np.concatenate((v0,[1]))) [0:3]
    v1 = cc1 - np.matmul(WtoIs[0], np.matmul(ItoWs[i+1], np.asarray(points1[idx])).T)[0:3]
    v1 = np.matmul(ItoWs[i+1], np.concatenate((v1,[1]))) [0:3]
    #print(v0, v1)
    tmp1 = ((eye - np.matmul(v0, v0.T)) + (eye - np.matmul(v1, v1.T))).T
    tmp2 = np.matmul((eye - np.matmul(v0, v0.T)), cc0) + np.matmul((eye - np.matmul(v1, v1.T)), cc1)
    point = np.matmul(tmp1, tmp2)
    
    res_points.append(point)

#print(len(res_points))
print(res_points)
xs = [p[0] for p in res_points]
ys = [p[1] for p in res_points]
zs = [p[2] for p in res_points]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs, ys, zs, zdir='z', s=20)
plt.show()
