import cv2
import numpy as np


def show_img(filename, scale=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(filename, scale)

    #change_channel(img)
    write_rectangle(img)
    image_properties(img)

    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def change_channel(img):
    height, width, depth = np.shape(img)
    for py in range(0, height):
        for px in range(0, width):
            img[py][px][0] = 0


def write_rectangle(img):
    # The parameters here are the image, the top left coordinate, bottom right coordinate, color, and line thickness.
    cv2.rectangle(img, (250, 40), (700, 490), (0, 0, 255), 5)
    cv2.putText(img, 'Specjal', (270, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (200, 255, 155), 2, cv2.LINE_AA)


def image_properties(img):
    print("The properties of the image are:")
    print("Shape:" + str(img.shape))
    print("Total no. of pixels:" + str(img.size))
    print("Data type of image:" + str(img.dtype))


def main():
    show_img("examples/specjal.jpg")
    # change_channel()


if __name__ == "__main__":
    main()
