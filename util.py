from scipy import misc
import numpy as np

"""
   Convert an RGB image to grayscale.

   This function applies a fixed weighting of the color channels to form the
   resulting intensity image.

   Arguments:
      rgb   - a 3D numpy array of shape (sx, sy, 3) storing an RGB image

   Returns:
      gray  - a 2D numpy array of shape (sx, sy) storing the corresponding
              grayscale image
"""
def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
    return gray

"""
   Load an image and convert it to grayscale.

   Arguments:
      filename - image file to load

   Returns:
      image    - a 2D numpy array containing a grayscale image
"""
def load_image(filename):
   image = misc.imread(filename)
   image = image / 255;
   if (image.ndim == 3):
      image = rgb2gray(image)
   return image

"""
  pts1 and pts2 are world position of interest points. they have dimension of n * 3
  Remember: change index space points to world position before calling this function
"""
def computeFundamentalMatrix(pts1, pts2):
  n = len(pts1)
  
  if len(pts2) != n:
    raise ValueError("imcompatible number of points")

  M = np.zeros((n,9))
  for i in range(n):
    pt1 = pts1[i]
    pt2 = pts2[i]
    M[i] = [pt1[j] * pt2[k] for j in range(3) for k in range(3)]
  U, S, V = np.linalg.svd(M)
  fundamentalMatrix = V[-1].reshape(3,3)
  U, S, V = np.linalg.svd(fundamentalMatrix)
  S[2] = 0
  fundamentalMatrix = np.dot(U, np.dot(np.diag(S), V))

  return fundamentalMatrix/fundamentalMatrix[2,2] # build homogenous matrix

