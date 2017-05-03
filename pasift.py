#!/usr/bin/env python

'''
Affine invariant feature-based image matching sample.

This sample is similar to find_obj.py, but uses the affine transformation
space sampling technique, called ASIFT [1]. While the original implementation
is based on SIFT, you can try to use SURF or ORB detectors instead. Homography RANSAC
is used to reject outliers. Threading is used for faster affine sampling.

[1] http://www.ipol.im/pub/algo/my_affine_sift/

USAGE
  asift.py [--feature=<sift|surf|orb|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

# Python 2/3 compatibility
from __future__ import print_function

from random import randint

import numpy as np
import cv2
import json
from joblib import Parallel, delayed
import multiprocessing

# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

# local modules
from imutils.perspective import four_point_transform

from common import Timer
from find_obj import init_feature, filter_matches, explore_match


def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i + 1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)


if __name__ == '__main__':
    print(__doc__)

    import sys, getopt

    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'brisk-flann')
    fn = []
    # try:
    # fn1 = args
    # except:
    fn1 = 'pic/frame0.jpg'
    fn.append('pic/frame278.jpg')
    fn.append('pic/frame76.jpg')
    fn.append('pic/frame110.jpg')
    fn.append('pic/frame120.jpg')
    fn.append('pic/frame410.jpg')
    fn.append('pic/frame305.jpg')
    fn.append('pic/frame330.jpg')
    fn.append('pic/frame255.jpg')
    img1 = cv2.imread(fn1)
    img = []
    for f in fn:
        img.append(cv2.imread(f))
    detector, matcher = init_feature(feature_name)
    with open('pic/task_data.json', 'r') as fp:
        corners = json.load(fp)

    rect = np.array([[corners['object_coord_in_ref_frame']['top_right']['x'],
                      corners['object_coord_in_ref_frame']['top_right']['y']],
                     [corners['object_coord_in_ref_frame']['top_left']['x'],
                      corners['object_coord_in_ref_frame']['top_left']['y']],
                     [corners['object_coord_in_ref_frame']['bottom_left']['x'],
                      corners['object_coord_in_ref_frame']['bottom_left']['y']],
                     [corners['object_coord_in_ref_frame']['bottom_right']['x'],
                      corners['object_coord_in_ref_frame']['bottom_right']['y']]])
    print(rect)
    # img1 = four_point_transform(img1, rect)
    cv2.imwrite('img.jpg', img1)
    if img1 is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    # if img2 is None:
    #     print('Failed to load fn2:', fn2)
    #     sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    print('using', feature_name)

    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, img1, pool=pool)
    kp = []
    desc = []
    for i in img:
        kpt, desct = affine_detect(detector, i, pool=pool)
        kp.append(kpt)
        desc.append(desct)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp)))


    def match_and_draw(kp2, desc2, img2,c):
        # with Timer('matching'):
        print('matching ')
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p2, p1, cv2.RANSAC, 5.0)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
            # do not draw outliers (there will be a lot of them)
            kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        else:
            H, status = None, None
            print('%d matches found, not enough for homography estimation' % len(p1))
        heightB, widthB = img2.shape[:2]

        im_dst = cv2.warpPerspective(img2, H, (widthB, heightB))
        cv2.imwrite('test' + str(c) + '.jpg', im_dst)
        print('finish')
        return 'finish'


    def match_wrap(args):
        # args = list(args)
        match_and_draw(*args)


    def calb():
        print('pass')
        pass


    # pool = multiprocessing.Pool()
    c = 0
    for kp2, desc2, img2 in zip(kp, desc, img):
        c = c + 1
        my_list = [kp1.copy(), kp2, desc1.copy(), desc2, img2, c]
        # p = multiprocessing.Process(target=match_and_draw, args=(kp1.copy(), kp2, desc1.copy(), desc2, img2, c))
        # p.start()
        # p.join()
        # results = pool.apply_async(match_and_draw, args=(kp1.copy(), kp2, desc1.copy(), desc2, img2, c),callback=calb)
        # results = pool.apply_async(match_and_draw, my_list)
        # results.wait()
    # nums = [1, 2, 3, 4, 5, 6, 7, 8]
    results = [pool.apply_async(match_and_draw, args=(kp2, desc2, img2, c)) for kp2, desc2, img2, c in
               zip(kp, desc, img, range(1, 111))]
    output = [p.get() for p in results]
    # print(output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
