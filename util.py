from scipy import misc
import numpy as np
 
def rgb2gray(rgb):
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
    gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
    return gray

def load_image(filename):
    """
    Load an image and convert it to grayscale.

    Arguments:
        filename - image file to load

    Returns:
        image    - a 2D numpy array containing a grayscale image
    """
    image = misc.imread(filename)
    image = image / 255
    if (image.ndim == 3):
        image = rgb2gray(image)
    return image

def compute_fundamental(pts1, pts2):
    """
    pts1 and pts2 are world position of interest points. they have dimension of n * 3
    Remember: change index space points to world position before calling this function
    """
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

def compute_P2(F):
    """
    Assuming P1 = [I 0], compute the second camera projection matrix from a fundamental matrix. 
    """ 
    # compute left epipole
    _, _, V = np.linalg.svd(F.T)
    tmp = V[-1]
    le = tmp / tmp[2]

    # Skew matrix so that a x v = Av for any v.
    a = np.array([[0, -le[2], le[1]], [le[2], 0, -le[0]], [-le[1], le[0], 0]])
    return np.vstack((np.dot(a, F.T).T, le)).T
