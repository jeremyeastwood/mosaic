import cv2
import os
import numpy as np


def get_features(im, alg='SIFT'):
    if alg == 'SIFT':
        detector = cv2.SIFT()
    elif alg == 'ORB':
        detector = cv2.ORB()
    elif alg == 'SURF':
        detector = cv2.SURF()

    return detector.detectAndCompute(im, None)


def draw_keypoints(im, keypoints, outfile):
    im = cv2.drawKeypoints(im, keypoints, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(outfile, im)
    os.system("open {}".format(outfile))


def find_blobs(im):
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10

    # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.5

    # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87

    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.1

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # detector = cv2.SimpleBlobDetector()

    # Detect blobs.
    blobs = detector.detect(im)

    return blobs


def blur(im):
    return cv2.GaussianBlur(im, (5, 5), 0)


def warpTwoImages(img1, img2, H):
    '''warp img1 to img2 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts1_ = cv2.perspectiveTransform(pts1, H)
    pts = np.concatenate((pts2, pts1_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img2
    return result


def combine_images(im1, im2, H12, H2, size):
    w, h = size
    h2, w2 = im2.shape[:2]
    if H2 is None:
        H2 = np.float32([[1, 0, w/2-w2/2], [0, 1, h/2-h2/2], [0, 0, 1]])
    warped_im2 = cv2.warpPerspective(im2, H2, size)
    warped_im1 = cv2.warpPerspective(im1, H2.dot(H12), size)
    # cv2.imwrite("warped_im2.jpg", warped_im2)
    # cv2.imwrite("warped_im1.jpg", warped_im1)
    mask = cv2.cvtColor(warped_im2, cv2.COLOR_BGR2GRAY) != 0
    warped_im1[mask] = 0
    return warped_im1 + warped_im2
    # cv2.imwrite("warped_added.jpg", cv2.add(warped_im1, warped_im2, mask=mask))

    # alpha = 0.5
    # beta = (1.0 - alpha)
    # blended = cv2.addWeighted(warped_im2, alpha, warped_im1, beta, 0.0)
    # cv2.imwrite("blended.jpg", blended)

    # warped_im1[h-h2/2:h+h2/2, w-w2/2:w+w2/2] = im2[:, :]
    # return warped_im1


def generate_warped_image(im, H, size):
    return cv2.warpPerspective(im, H, size)


def add_two_images(im1, im2):
    mask = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) != 0
    im1[mask] = 0
    return im1 + im2


def get_perspective_transform(image_size, pixel_coords):
    # src = np.float32([[0, 0], [0, image_size[0]], [image_size[1], image_size[0]], [image_size[1], 0]])
    src = np.float32([[0, 0], [image_size[0], 0], [image_size[0], image_size[1]], [0, image_size[1]]])
    print src
    ordered_coords = [pixel_coords[0], pixel_coords[3], pixel_coords[2], pixel_coords[1]]  # clockwise from bottom left
    # ordered_coords = map(lambda x: x[::-1], ordered_coords)
    dst = np.float32(ordered_coords)
    print dst
    return cv2.getPerspectiveTransform(src, dst)


def drawMatches(img1, kp1, img2, kp2, matches, vertical=False):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    if vertical:
        out = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
        # Place the first image at the top
        out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
        # Place the next image below it
        out[rows1:rows1+rows2, :cols2, :] = np.dstack([img2, img2, img2])

    else:
        out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')
        # Place the first image to the left
        out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
        # Place the next image to the right of it
        out[:rows2, cols1:cols1+cols2, :] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        px1, py1 = int(x1), int(y1)
        if vertical:
            px2, py2 = int(x2), int(y2) + rows1
        else:
            px2, py2 = int(x2) + cols1, int(y2)

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour red
        # thickness = 1
        cv2.circle(out, (px1, py1), 4, (0, 0, 255), 1)
        cv2.circle(out, (px2, py2), 4, (0, 0, 255), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (px1, py1), (px2, py2), (255, 0, 0), 1)

    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out