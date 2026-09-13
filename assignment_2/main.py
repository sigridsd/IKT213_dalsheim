import cv2
import numpy as np

def padding(image, border_width):
    padded = cv2.copyMakeBorder(
        image,
        border_width, border_width, border_width, border_width,
        cv2.BORDER_REFLECT
    )
    cv2.imwrite("solutions/padded_reflect.png", padded)
    return padded

def crop(image, x_0, x_1, y_0, y_1):
    cropped = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("solutions/cropped.png", cropped)
    return cropped

def resize(image, width, height):
    resized = cv2.resize(image, (width, height))
    cv2.imwrite("solutions/resized.png", resized)
    return resized

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]
    cv2.imwrite("solutions/copied.png", emptyPictureArray)
    return emptyPictureArray

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("solutions/grayscale.png", gray)
    return gray

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("solutions/hsv.png", hsv_image)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = (int(image[y, x, c]) + hue) % 256
    cv2.imwrite("solutions/hue_shifted.png", emptyPictureArray)
    return emptyPictureArray

def smoothing(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite("solutions/smoothing.png", blurred)
    return blurred


def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)

    cv2.imwrite("solutions/rotation.png", rotated)
    return rotated

def main():
    img = cv2.imread("iris-1.jpg")
    padding(img, 100)

    height, width, _ = img.shape
    crop(img, 200, width - 130, 200, height - 130)

    resize(img, 200, 200)

    height, width, channels = img.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copy(img, emptyPictureArray)

    grayscale(img)

    hsv(img)

    emptyPictureArray2 = np.zeros((height, width, 3), dtype=np.uint8)
    hue_shifted(img, emptyPictureArray2, 50)

    smoothing(img)

    rotation(img, 180)

main()