'''
    match_image module:

    Given two OpenCV images A and T (template),
    returns degree of confidence for whether A is in T.

    Major dependencies:
    OpenCV-python as cv2
    numpy as np

    Created by Ian Arawjo, Aug 7 2015
'''
import numpy as np
import cv2

# debug -- benchmarking
from timer import Timer

def resize(img, scale):
    return cv2.resize(img, (0,0), fx=scale, fy=scale)

# Container class for match return value.
# Stores homography, a table of confidence percentages, etc
#class Match(object):

class FeatureSet(object):
    def __init__(self, img, name):
        self.kp, self.desc = cv2.SIFT().detectAndCompute(img, None)
        self.name = name

# TBI: Does the same as batchMatch, except ...
#def batchMatchImage(templates, targetImages):

# Given a list of template features ([FeatureSet])
# and one target feature,
# output the best match as a tuple (templateName, perspectiveTransform, numOfMatchingFeatures).
# *Outputs None if no match was found.
def batchMatch(templates, target):

    # Create features. For each
    # target image, determine what template
    # it matches. Note that we could load all the features
    # at once before processing; however, this sucks up memory.

    # Compare feature set to each template
    # to find the best match.
    bestMatch = None
    for template in templates:
        matchingPairs, H = match_features(template, target, detector, matcher)
        if H is not None:
            # Match found. Replace if better than previous found match:
            if bestMatch is None or bestMatch[2] < matchingPairs:
                bestMatch = (template.name, H, matchingPairs)

    return bestMatch

def match(template, target, templateName, targetName):

    # Helper function
    def init_SIFT():
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)
        return detector, matcher

    # Init SIFT feature detector + matcher
    with Timer('Loading SIFT') as t:
        detector, matcher = init_SIFT()

    # Generate (keypoint, descriptor) features for the images
    templateFeatures = FeatureSet(template, templateName)

    #templateKp, templateDesc = detector.detectAndCompute(template, None)
    with Timer('Computing features') as t:
        targetFeatures = FeatureSet(target, targetName)

    # Given the features (keypoints, descriptors), a matcher, and a detector,
    # perform matching with OpenCV.
    match_features(templateFeatures, targetFeatures, detector, matcher)

def match_features(templateFeatures, targetFeatures, detector, matcher):

    templateKp, templateDesc = (templateFeatures.kp, templateFeatures.desc)
    targetKp, targetDesc = (targetFeatures.kp, targetFeatures.desc)

    # Match image features by k-Nearest-Neighbor algo
    with Timer('kNN matching') as t:
        matches = matcher.knnMatch(templateDesc, targetDesc, k = 2)

    # Apply ratio test from D. Lowe
    with Timer('Ratio filtering') as t:
        good = []
        def filter_matches(kp1, kp2, matches, ratio = 0.75):
            mkp1, mkp2 = [], []
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                    m = m[0]
                    mkp1.append( kp1[m.queryIdx] )
                    mkp2.append( kp2[m.trainIdx] )
            p1 = np.float32([kp.pt for kp in mkp1])
            p2 = np.float32([kp.pt for kp in mkp2])
            kp_pairs = zip(mkp1, mkp2)
            return p1, p2, kp_pairs
        p1, p2, good_pairs = filter_matches(templateKp, targetKp, matches)

    # DOES NOT WORK IN OPENCV FOR PYTHON <3
    # cv2.drawMatchesKnn expects list of lists as matches.
    #img3 = cv2.drawMatchesKnn(template,templateKp,target,targetKp,good,flags=2)
    #plt.imshow(img3),plt.show()

    with Timer('Computing homography') as t:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

    if H is not None:
        print H

        # debug -- Display match in window.
        explore_match('match', template, target, good_pairs, status, H)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # Found match. Return number of matching features.
        return (len(good_pairs), H)
    else:
        return (0, None)

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    perspective_corrected_img = cv2.warpPerspective(img2, np.linalg.inv(H), (w1, h1))
    vis2 = cv2.adaptiveThreshold(perspective_corrected_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #vis2 = resize(vis2 - img1, 0.5)

    # Remove salt-and-pepper noise
    #kernel = np.ones((3,3),np.uint8)
    #vis2 = cv2.morphologyEx(vi, cv2.MORPH_OPEN, kernel)

    vis2 = cv2.bitwise_not( cv2.bitwise_and(cv2.bitwise_not( vis2 ), img1))

    #vis2 = cv2.medianBlur(vis2, 3)
    #kernel = np.ones((3,3),np.uint8)
    #vis2 = cv2.morphologyEx(vis2, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Perspective correction: ', vis2)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))
        print "Found match."

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    vis = cv2.resize(vis, (0,0), fx=0.3, fy=0.3)

    cv2.imshow(win, vis)
    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                 (x1, y1), (x2, y2) = p1[i], p2[i]
                 col = (red, green)[status[i]]
                 cv2.line(cur_vis, (x1, y1), (x2, y2), col)
                 kp1, kp2 = kp_pairs[i]
                 kp1s.append(kp1)
                 kp2s.append(kp2)
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, flags=4, color=kp_color)

        cv2.imshow(win, cur_vis)
    cv2.setMouseCallback(win, onmouse)
    return vis

# (DEBUG) If run from command line --
if __name__ == '__main__':

    import sys, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('templateImgPath', action='store')
    parser.add_argument('targetImgPath', action='store')
    args = parser.parse_args()

    try:
        templateImgPath = args.templateImgPath
        targetImgPath = args.targetImgPath
    except:
        print "Error: The matcher requires two arguments: a template image file, and a target image file, in that order."
        sys.exit(0)

    # Attempt to read images
    try:
        template = cv2.imread(templateImgPath,0)
    except:
        print "Error: Could not find template image at path " + templateImgPath
        sys.exit(0)

    try:
        target = cv2.imread(targetImgPath, 0)
    except:
        print "Error: Could not find target image at path " + targetImgPath
        sys.exit(0)

    # Resize if necessary...
    template = resize(template, 0.4)
    target = resize(target, 0.4)

    match(template, target, templateImgPath, targetImgPath)
