import cv2

def print_image_information(image):
    height, width, channels = image.shape
    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("size:", image.size)
    print("data type:", image.dtype)

def save_camera_information():
    cam = cv2.VideoCapture(0)
    fps = cam.get(cv2.CAP_PROP_FPS)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cam.release()

    with open("solutions/camera_outputs.txt", "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

def main():
    img = cv2.imread("iris-1.jpg")
    print_image_information(img)
    save_camera_information()

if __name__ == "__main__":
    main()