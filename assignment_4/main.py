import cv2
import numpy as np


def harris(reference_image):
    img = reference_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite("solutions/harris.png", img)
    return img

def align_images(image_to_align, reference_image, max_features, good_match_precent):
    gray_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_align, None)
    kp2, des2 = sift.detectAndCompute(gray_ref, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    if len(good) < max_features:
        print(f"Ikke nok matcher: {len(good)}/{max_features}")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image_to_align, H, (w, h))

    matches_img = cv2.drawMatches(
        image_to_align, kp1, reference_image, kp2, good, None,
        matchesMask=mask.ravel().tolist(), flags=2
    )
    print(matches_img.shape)

    cv2.imwrite("solutions/aligned.png", aligned)
    matches_img = cv2.resize(matches_img, None, fx=0.5, fy=0.5)
    cv2.imwrite("solutions/matches.png", matches_img)
    return aligned


def main():
    reference = cv2.imread("reference_img.png")
    harris(reference)

    to_align = cv2.imread("align_this.jpg")
    align_images(to_align, reference, 10, 0.7)

main()