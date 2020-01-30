import cv2


def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


def record_video(filename, fps=20, resolution=(640, 480)):
    # 0,1 is recording from the camera
    cap = cv2.VideoCapture(list_cameras()[-1])

    # To set the resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # FourCC is a 4-byte code used to specify the video codec.
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 Encoder. Compresses video to the H.264 format.
    out = cv2.VideoWriter(filename, fourcc, fps, resolution)

    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame by frame
        out.write(frame)
        cv2.imshow('Original', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Close the window / Release webcam
    out.release()  # After we release our webcam, we also release the output
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


def main():
    record_video("output/nagranie.mp4")


if __name__ == "__main__":
    main()
