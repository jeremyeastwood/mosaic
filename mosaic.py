import cv2, sys, glob, os
from itertools import combinations
import numpy as np
import image_utils
from image_projections import get_image_location, plot_footprints


class Image(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.directory = os.path.dirname(filepath)
        self.rootname = self.filename.split('.')[0]
        self.keypoints = None
        self.descriptors = None
        self.blobs = None
        self.image_location = None

    @property
    def data(self):
        return cv2.imread(self.filepath)

    @property
    def grayscale(self):
        return cv2.imread(self.filepath, 0)

    def get_features(self, alg='SIFT'):
        self.keypoints, self.descriptors = image_utils.get_features(self.grayscale, alg=alg)

    def get_blobs(self):
        self.blobs = image_utils.find_blobs(image_utils.blur(self.grayscale))

    def get_image_location(self):
        self.image_location = get_image_location(self.filepath)


def get_images(file_list):
    images = glob.glob(file_list)
    return [Image(im) for im in images]


def get_pairs(images):
    return [Pair(im1, im2) for (im1, im2) in list(combinations(images, 2))]


class Pair(object):
    def __init__(self, im1, im2):
        self.im1 = im1
        self.im2 = im2
        self.matches = None
        self.homography = None

    def get_matches(self, alg='BF', plot=False):

        if alg == 'BF':
            # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # matches = matcher.match(im1.descriptors, im2.descriptors)
            # matches = sorted(matches, key=lambda x: x.distance)
            matcher = cv2.BFMatcher()
        elif alg == 'FLANN':
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)   # or pass empty dictionary
            matcher = cv2.FlannBasedMatcher(index_params, search_params)

        matches = matcher.knnMatch(self.im1.descriptors, self.im2.descriptors, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        if plot:
            pic = os.path.join('matches.jpg')
            im3 = image_utils.drawMatches(
                self.im1.grayscale,
                self.im1.keypoints,
                self.im2.grayscale,
                self.im2.keypoints,
                good,
                vertical=True
            )
            cv2.imwrite(pic, im3)
            os.system("open {}".format(pic))

        self.matches = good

    def get_homography(self):
        MIN_MATCH_COUNT = 4

        if len(self.matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([self.im1.keypoints[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.im2.keypoints[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)

            # print src_pts
            # print dst_pts

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            self.homography = M

            # print M
            # print mask
            # matchesMask = mask.ravel().tolist()

            # h, w, b = self.im1.data.shape
            # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            # dst = cv2.perspectiveTransform(pts, M)
            # # print dst

            # mosaic = cv2.polylines(self.im2.data, [np.int32(dst)], False, 255, 3, cv2.CV_AA)
            # cv2.imwrite('mosaic.jpg', mosaic)

        else:
            print "Not enough matches are found - %d/%d" % (len(self.matches), MIN_MATCH_COUNT)
            # matchesMask = None

        # print matchesMask

    def generate_mosaic(self):
        # mosaic = image_utils.warpTwoImages(self.im1.data, self.im2.data, self.homography)
        h1, w1 = self.im1.data.shape[:2]
        h2, w2 = self.im2.data.shape[:2]
        h, w = map(max, [[h1, h2], [w1, w2]])
        H2 = np.float32([[1, 0, w-w2/2], [0, 1, h-h2/2], [0, 0, 1]])
        mosaic = image_utils.combine_images(self.im1.data, self.im2.data, self.homography, H2, (2*w, 2*h))
        cv2.imwrite("warped_mosaic.jpg", mosaic)


def get_canvas(images):
    lngs = []
    lats = []
    for im in images:
        intersections = im.image_location.intersections
        points = [p.cartesian_point for p in intersections]
        x, y = zip(*points)
        lngs.extend(x)
        lats.extend(y)
    canvas = [
        (min(lngs), min(lats)),
        (min(lngs), max(lats)),
        (max(lngs), max(lats)),
        (max(lngs), min(lats)),
    ]
    print "canvas: dx={}, dy={}".format(max(lngs) - min(lngs), max(lats) - min(lats))
    return canvas


if __name__ == '__main__':

    import optparse
    parser = optparse.OptionParser()
    parser.add_option("--input", dest="input",  type='str', help="input images (required)")
    parser.add_option("--output", dest="output",  type='str', help="output directory (required)")
    opts, args = parser.parse_args()

    if not opts.input or not opts.output:
        parser.print_help()
        sys.exit(1)

    images = get_images(opts.input)
    # for im in images:
    #     im.get_blobs()
    #     print im.filename, len(im.blobs)
    #     outfile = os.path.join(im.directory, im.rootname + '_blobs.jpg')
    #     image_utils.draw_keypoints(im.data, im.blobs, outfile)

    for im in images:
        im.get_image_location()
        print im.image_location.footprint_size
    canvas = get_canvas(images)
    plot_footprints([im.image_location for im in images])

    for im in images:
        im.get_features()
        print im.filename, len(im.keypoints)
        # image_utils.draw_keypoints(im.data, im.keypoints, os.path.join(im.directory, im.rootname + '_keys.jpg'))

    pairs = get_pairs(images)
    print len(pairs)
    for pair in pairs:
        pair.get_matches(plot=False)
        pair.get_homography()
        pair.generate_mosaic()

    # for pair in sorted(pairs, key=lambda x: len(x.matches), reverse=True):
    #     print pair.im1.filename, pair.im2.filename, len(pair.matches)

