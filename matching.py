import numpy as np
from canny import sobel_gradients, conv_2d_gaussian, mirror_border, trim_border, pad_border
  
def find_interest_points(image, max_points = 200, scale = 1.0):
    """
    INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)
 
    Implement an interest point operator of your choice.
 
    Your operator could be:
 
    (A) The Harris corner detector (Szeliski 4.1.1)
 
                OR
 
    (B) The Difference-of-Gaussians (DoG) operator defined in:
        Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
 
                OR
 
    (C) Any of the alternative interest point operators appearing in
        publications referenced in Szeliski or in lecture
 
               OR
 
    (D) A custom operator of your own design
 
    You implementation should return locations of the interest points in the
    form of (x,y) pixel coordinates, as well as a real-valued score for each
    interest point.  Greater scores indicate a stronger detector response.
 
    In addition, be sure to apply some form of spatial non-maximum suppression
    prior to returning interest points.
 
    Whichever of these options you choose, there is flexibility in the exact
    implementation, notably in regard to:
 
    (1) Scale
 
        At what scale (e.g. over what size of local patch) do you operate?
 
        You may optionally vary this according to an input scale argument.
 
        We will test your implementation at the default scale = 1.0, so you
        should make a reasonable choice for how to translate scale value 1.0
        into a size measured in pixels.
 
    (2) Nonmaximum suppression
 
        What strategy do you use for nonmaximum suppression?
 
        A simple (and sufficient) choice is to apply nonmaximum suppression
        over a local region.  In this case, over how large of a local region do
        you suppress?  How does that tie into the scale of your operator?
 
    For making these, and any other design choices, keep in mind a target of
    obtaining a few hundred interest points on the examples included with
    this assignment, with enough repeatability to have a large number of
    reliable matches between different views.
 
    If you detect more interest points than the requested maximum (given by
    the max_points argument), return only the max_points highest scoring ones.
 
    In addition to your implementation, include a brief write-up (in hw2.pdf)
    of your design choices.
 
    Arguments:
       image       - a grayscale image in the form of a 2D numpy array
       max_points  - maximum number of interest points to return
       scale       - (optional, for your use only) scale factor at which to
                     detect interest points
 
    Returns:
       xs          - numpy array of shape (N,) containing x-coordinates of the
                     N detected interest points (N <= max_points)
       ys          - numpy array of shape (N,) containing y-coordinates
       scores      - numpy array of shape (N,) containing a real-valued
                     measurement of the relative strength of each interest point
                     (e.g. corner detector criterion OR DoG operator magnitude)
    """
    # check that image is grayscale
    assert image.ndim == 2, 'image should be grayscale'

    dx, dy = sobel_gradients(image)
    Ix2 = conv_2d_gaussian(dx ** 2)
    Iy2 = conv_2d_gaussian(dy ** 2)
    IxIy = conv_2d_gaussian(dx * dy)
    # measured interest = determinant over trace, by Brown et al 2005
    interest = (Ix2 * Iy2 - IxIy ** 2) / (Ix2 + Iy2)
    np.seterr(divide='ignore', invalid='ignore')
    # now apply nonmaximum suppression over a local 11*11 region for scale 1
    half = int(2 * scale + 3)
    width = half * 2 + 1
    interest = mirror_border(interest, half, half)
    nonmax_suppressed = np.zeros(interest.shape)
    for i in range(half, interest.shape[0] - half, width):
        for j in range(half, interest.shape[1] - half, width):
            window = interest[i - half:i + half + 1, j - half:j + half + 1]
            val = np.max(window)
            ind = np.unravel_index(np.argmax(window), window.shape)
            nonmax_suppressed[i - half + ind[0]][j - half + ind[1]] = val
    nonmax_suppressed = trim_border(nonmax_suppressed, half, half)
 
    # choose the first max_points of interest values if there are more than enough
    xs, ys = np.where(nonmax_suppressed >  0.0001)
    if len(xs) > max_points:
       xs, ys = np.unravel_index(np.argpartition(nonmax_suppressed.ravel(), -max_points)[-max_points:], nonmax_suppressed.shape)
    scores = nonmax_suppressed[xs, ys]

    return xs, ys, scores

def extract_features(image, xs, ys, scale = 1.0):
    """
    FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)
 
    Implement a SIFT-like feature descriptor by binning orientation energy
    in spatial cells surrounding an interest point.
 
    Unlike SIFT, you do not need to build-in rotation or scale invariance.
 
    A reasonable default design is to consider a 3 x 3 spatial grid consisting
    of cell of a set width (see below) surrounding an interest point, marked
    by () in the diagram below.  Using 8 orientation bins, spaced evenly in
    [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.
 
              ____ ____ ____
             |    |    |    |
             |    |    |    |
             |____|____|____|
             |    |    |    |
             |    | () |    |
             |____|____|____|
             |    |    |    |
             |    |    |    |
             |____|____|____|
 
                  |----|
                   width
 
    You will need to decide on a default spatial width.  Optionally, this can
    be a multiple of a scale factor, passed as an argument.  We will only test
    your code by calling it with scale = 1.0.
 
    In addition to your implementation, include a brief write-up (in hw2.pdf)
    of your design choices.
 
    Arguments:
       image    - a grayscale image in the form of a 2D numpy
       xs       - numpy array of shape (N,) containing x-coordinates
       ys       - numpy array of shape (N,) containing y-coordinates
       scale    - scale factor
 
    Returns:
       feats    - a numpy array of shape (N,K), containing K-dimensional
                  feature descriptors at each of the N input locations
                  (using the default scheme suggested above, K = 72)
    """
    # check that image is grayscale
    assert image.ndim == 2, 'image should be grayscale'
    ##########################################################################
    width = int(scale * 2 + 1)
    half = (width - 1) // 2
    image = pad_border(image, width, width)
    dx, dy = sobel_gradients(image)
    PI = np.pi
    n = len(xs)
    feats = np.zeros((n, 72))
    for point in range(n):
        xi = xs[point]
        yi = ys[point]
        for i in range(3):
            for j in range(3):
                px = xi + (j - 1) * width
                py = yi + (i - 1) * width
                for xx in range(-half, half + 1):
                    for yy in range(-half, half + 1):
                        x = dx[px + xx, py + yy]
                        y = dy[px + xx, py + yy]
                        angle = np.arctan2(y, x)
                        index = int(angle / (PI / 4.0)) if int(angle / (PI / 4.0)) >= 0 else 8 - int(angle / (PI / 4.0))
                        feats[point][i * j * 8 + index] += np.sqrt(x ** 2 + y ** 2)
    ##########################################################################
    return feats


def match_features(feats0, feats1, scores0, scores1):
    """
    FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)
 
    Given two sets of feature descriptors, extracted from two different images,
    compute the best matching feature in the second set for each feature in the
    first set.
 
    Matching need not be (and generally will not be) one-to-one or symmetric.
    Calling this function with the order of the feature sets swapped may
    result in different returned correspondences.
 
    For each match, also return a real-valued score indicating the quality of
    the match.  This score could be based on a distance ratio test, in order
    to quantify distinctiveness of the closest match in relation to the second
    closest match.  It could optionally also incorporate scores of the interest
    points at which the matched features were extracted.  You are free to
    design your own criterion.
 
    In addition to your implementation, include a brief write-up (in hw2.pdf)
    of your design choices.
 
    Arguments:
       feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                  feature descriptors (generated via extract_features())
       feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                  feature descriptors (generated via extract_features())
       scores0  - a numpy array of shape (N0,) containing the scores for the
                  interest point locations at which feats0 was extracted
                  (generated via find_interest_point())
       scores1  - a numpy array of shape (N1,) containing the scores for the
                  interest point locations at which feats1 was extracted
                  (generated via find_interest_point())
 
    Returns:
       matches  - a numpy array of shape (N0,) containing, for each feature
                  in feats0, the index of the best matching feature in feats1
       scores   - a numpy array of shape (N0,) containing a real-valued score
                  for each match
    """
    ##########################################################################
    n0 = len(feats0)
    n1 = len(feats1)
    matches = np.zeros((n0,1), dtype = int)
    scores = np.zeros((n0,1))
    for i in range(n0):
        d0 = np.sqrt(np.sum((feats0[i] - feats1[0]) ** 2))
        d1 = np.sqrt(np.sum((feats0[i] - feats1[1]) ** 2))
        if d0 < d1:
            min_dist = d0
            best_id = 0
            second_dist = d1
            second_id = 1
        else:
            min_dist = d1
            best_id = 1
            second_dist = d0
            second_id = 0
        for j in range(2, n1):
            matching = np.sqrt(np.sum((feats0[i] - feats1[j]) ** 2))
            if matching < min_dist:
                second_dist = min_dist
                second_id = best_id
                min_dist = matching
                best_id = j
        # score is using inverse of distance ratio test, so the higher the better
        matches[i] = best_id
        scores[i] = second_dist / min_dist
    #print(matches)
    ##########################################################################
    return matches, scores

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""
from collections import defaultdict
def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   ##########################################################################
   n0 = len(xs0)
   votes = defaultdict(float)
   for i in range(n0):
      x = int(xs0[i] - xs1[matches[i]])
      y = int(ys0[i] - ys1[matches[i]])
      votes[(x,y)] += scores[i]
   best = max(votes.items(), key = lambda k: k[1])
   tx, ty = best[0]
   ##########################################################################
   return tx, ty, votes

def find_matches(img0, img1):
    N = 150
    xs0, ys0, scores0 = find_interest_points(img0, N, 2.5)
    xs1, ys1, scores1 = find_interest_points(img1, N, 2.5)
    feats0 = extract_features(img0, xs0, ys0, 2.0)
    feats1 = extract_features(img1, xs1, ys1, 2.0)
    matches, match_scores = match_features(feats0, feats1, scores0, scores1)
    threshold = 1.89 # adjust this for your match scoring system

    p1_match = []
    p2_match = []
    xm = xs1[matches]
    ym = ys1[matches]
    N = matches.size
    for n in range(N):
        if (match_scores[n] > threshold):
            p1_match.append([ys0[n], xs0[n], 0])
            p2_match.append([ym[n][0], xm[n][0], 0])
    return np.array(p1_match), np.array(p2_match)
