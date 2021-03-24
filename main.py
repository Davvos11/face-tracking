import cv2
from time import sleep

import pyfakewebcam as pyfakewebcam

CASC_PATH = "haarcascade_frontalface_default.xml"
INPUT_DEVICE = '/dev/video1'
OUTPUT_DEVICE = '/dev/video50'
WIDTH = 300
PAN_SPEED = 20


class CameraException(Exception):
    pass


def init_devices(input_device: str, output_device: str, output_width: int)\
        -> (cv2.VideoCapture, pyfakewebcam.FakeWebcam):
    # Open capture device
    video_capture = cv2.VideoCapture(input_device)
    # Check device
    if not video_capture.isOpened():
        raise CameraException('Unable to load camera.')

    # Capture one frame in order to get the resolution
    ret, frame = video_capture.read()
    shape = frame.shape

    # Open virtual output device
    video_output = pyfakewebcam.FakeWebcam(output_device, output_width, shape[0], shape[2])

    return video_capture, video_output


def pan(y_range: (int, int), y_target: (int, int)) -> (int, int):
    y1, y2 = y_range
    t1, t2 = y_target

    diff = abs(y1 - t1)

    if y1 > t1:
        y1 -= diff // PAN_SPEED
        y2 -= diff // PAN_SPEED
    elif y1 < t1:
        y1 += diff // PAN_SPEED
        y2 += diff // PAN_SPEED

    return y1, y2


def main():
    faceCascade = cv2.CascadeClassifier(CASC_PATH)

    video_capture, video_output = init_devices(INPUT_DEVICE, OUTPUT_DEVICE, WIDTH)
    y_range = (0, WIDTH)
    y_target = y_range

    i = 0

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if i % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                if len(faces) > 0:
                    input_width = frame.shape[1]
                    input_height = frame.shape[0]

                    # Get the crop range around the first face
                    (y, x, w, h) = faces[0]
                    center_y = y + (w // 2)
                    y_target = (center_y - WIDTH // 2, center_y + WIDTH // 2)

                    # cv2.circle(frame, (center_y, input_height // 2), 2, (0, 0, 255))
                    # cv2.rectangle(frame, (y_range[0], 0), (y_range[1], frame.shape[0]), (255, 0, 0), 2)
                    # cv2.rectangle(frame, (y, x), (y + w, x + h), (0, 255, 0), 2)

                    if y_target[0] < 0:
                        y_target = (0, WIDTH)
                    elif y_range[1] > input_width:
                        y_target = (input_width - WIDTH, input_width)

            # Display the resulting frame
            # cv2.imshow('Video', frame)

            # Get the new range based on the target
            y_range = pan(y_range, y_target)

            # Crop the resulting frame
            frame = frame[:, y_range[0]: y_range[1]]

            # Output to virtual device
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_output.schedule_frame(output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            i += 1

    except KeyboardInterrupt:
        pass
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
