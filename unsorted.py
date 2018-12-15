'''
def R_and_t_from_E(E, focal_length, lastRt, id, kp1, kp2, matches):
    """
    Estimate R and t from essential matrix
    """
    W = np.matrix([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]], dtype=np.float)

    U, D, V = np.linalg.svd(E)

    # two possibilities for R, t
    R1 = U * W * V.T
    R2 = U * W.T * V.T

    t1 = U[:, 2]
    t2 = -U[:, 2]

    # ensure positive determinants
    if np.linalg.det(R1) < 0:
        R1 = -R1

    if np.linalg.det(R2) < 0:
        R2 = -R2

    # extract match points
    matches1 = []
    matches2 = []
    for m in matches:
        pt1 = np.matrix([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]]).T
        matches1.append((pt1, m.queryIdx, frameIdx))

        pt2 = np.matrix([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]]).T
        matches2.append((pt2, m.trainIdx, frameIdx + 1))

    # create four possible new camera matrices
    Rt1 = np.hstack([R1, t1])
    Rt2 = np.hstack([R1, t2])
    Rt3 = np.hstack([R2, t1])
    Rt4 = np.hstack([R2, t2])

    # transform each Rt to be relative to baseRt
    baseRt4x4 = np.vstack([baseRt, np.matrix([0, 0, 0, 1], dtype=np.float)])
    Rt1 = Rt1 * baseRt4x4
    Rt2 = Rt2 * baseRt4x4
    Rt3 = Rt3 * baseRt4x4
    Rt4 = Rt4 * baseRt4x4

    # test how many points are in front of both cameras    
    bestRt = None
    bestCount = -1
    bestPts3D = None
    for Rt in [Rt1, Rt2, Rt3, Rt4]:

        cnt = 0
        pts3D = {}
        for m1, m2 in zip(matches1, matches2):

            # use least squares triangulation
            x = triangulateLM(baseRt, Rt, m1[0], m2[0], K)

            # test if in front of both cameras
            if inFront(baseRt, x) and inFront(Rt, x):
                cnt += 1
                pts3D[x.tostring()] = (m1, m2)
    
        # update best camera/cnt
        #print "[DEBUG] Found %d points in front of both cameras." % cnt
        if cnt > bestCount:
            bestCount = cnt
            bestRt = Rt
            bestPts3D = pts3D

    print ("Found %d of %d possible 3D points in front of both cameras.\n" % (bestCount, len(matches1)))

    # Wrap bestRt, bestPts3D into a 'pair'
    pair = {}
    pair["motion"] = [baseRt, bestRt]
    pair["3Dmatches"] = {}
    pair["frameOffset"] = frameIdx
    for X, matches in bestPts3D.items():
        m1, m2 = matches
        key = (m1[1], m1[2]) # use m1 instead of m2 for matching later

        entry = {"frames" : [m1[2], m2[2]], # frames that can see this point
                 "2Dlocs" : [m1[0], m2[0]], # corresponding 2D points
                 "3Dlocs" : X,              # 3D triangulation 
                 "newKey" : (m2[1], m2[2])} # next key (for merging with graph)
        pair["3Dmatches"][key] = entry
        
    return pair
'''



'''
res_points = []
for i in range(N):
    for j in range(i + 1, N):
        pts0 = points[i]
        pts1 = points[j]
        ret = triangulate(pts0, pts1, transformation_matrices[i], transformation_matrices[j])

print(ret)
'''



'''
def triangulate_point(x1,x2,P1,P2):
    """
    Point pair triangulation from least squares solution. 
    """
        
    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    _, _, V = np.linalg.svd(M)
    X = V[-1,:4]

    return X / X[3]


def triangulate(x1,x2,P1,P2):
    """
    Two-view triangulation of points in x1,x2 (3*n homogenous coordinates).
    """
        
    n = x1.shape[1]
    if x2.shape[1] != n:
        return "error"

    X = [triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return np.array(X)

def E_from_F(F, focal_length):
    '''
    compute Essential Matrix from Fundamental matrix and camera intrinsic matrix
    '''
    K = np.matrix([[focal_length, 0, 0],
                   [0, focal_length, 0],
                   [0, 0, focal_length]], dtype=np.float)
    return K.T * F * K
'''