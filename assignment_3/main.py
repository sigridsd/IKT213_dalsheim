import cv2
import numpy as np


def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel = cv2.convertScaleAbs(sobel)
    cv2.imwrite("solutions/sobel_edges.png", sobel)
    return sobel

def canny_edge_detection(image, threshold_1, threshold_2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, threshold_1, threshold_2)
    cv2.imwrite("solutions/canny_edges.png", canny)
    return canny

def template_match(image, template):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape

    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    locations = np.where(result >= threshold)

    matched = image.copy()
    for pt in zip(*locations[::-1]):
        cv2.rectangle(matched, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("solutions/template_match.png", matched)
    return matched

def resize(image, scale_factor: int, up_or_down: str):
    resized = image.copy()
    steps = int(np.log2(scale_factor))

    for _ in range(steps):
        if up_or_down == "up":
            resized = cv2.pyrUp(resized)
        elif up_or_down == "down":
            resized = cv2.pyrDown(resized)
        else:
            raise ValueError('up_or_down må være "up" eller "down"')

    cv2.imwrite(f"solutions/resized_{up_or_down}.png", resized)
    return resized

def main():
    img = cv2.imread("lambo.png")
    sobel_edge_detection(img)

    canny_edge_detection(img, 50, 50)

    shapes = cv2.imread("shapes-1.png")
    template = cv2.imread("shapes_template.jpg")
    template_match(shapes, template)

    resize(img, 2, "up")
    resize(img, 2, "down")
main()