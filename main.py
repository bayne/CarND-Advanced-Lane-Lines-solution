import numpy as np
import cv2
import os
import glob
import sys
import pickle


def calibrate(calibration_images, chessboard_shape, image_saver):
    """
    Generates the camera matrix to be used for distortion correction

    :param image_saver: ImageSaver
    :param calibration_images: An list of filenames to use as the calibration images
    :param chessboard_shape: an ordered pair that describes the number of corners in the chessboard
    :return: 
    """
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    objp = np.zeros((chessboard_shape[1] * chessboard_shape[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2)
    image_size = None

    for fname in calibration_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)
        if image_saver.enabled:
            filename = fname.split('/')[-1]
            img = cv2.drawChessboardCorners(image=img, patternSize=chessboard_shape, corners=corners,
                                            patternWasFound=ret)
            image_saver.save('calibrate', filename, img)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

        else:
            print('Could not found corners in calibration image: ' + fname, file=sys.stderr)

    return cv2.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints, imageSize=image_size, cameraMatrix=None,
                               distCoeffs=None)


class ImageSaver:
    def __init__(self, output_directory, enabled) -> None:
        self.enabled = enabled
        self.__output_directory = output_directory

    def save(self, sub_directory, filename, image):
        if self.enabled:
            directory = self.__output_directory + '/' + sub_directory + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(filename=directory + '/' + filename, img=image)


class Pipeline:
    def __init__(
            self,
            camera_matrix,
            dist_coeffs,
            image_saver,
            yellow_lane_hsv_range,
            white_lane_hsv_range
    ) -> None:
        self.__white_lane_hsv_range = white_lane_hsv_range
        self.__yellow_lane_hsv_range = yellow_lane_hsv_range
        self.__image_saver = image_saver
        self.__camera_matrix = camera_matrix
        self.__dist_coeffs = dist_coeffs
        self.current_filename = None

    # TODO get binary image
    # Color threshold
    # canny

    # TODO Perspective transform

    # TODO get lane pixels

    # TODO get lane curvature

    # TODO get vehicle position

    # TODO overlay lane indicator over source image

    # TODO visual numerical output of lane curvature and vehicle position

    def __undistort(self, image):
        image = cv2.undistort(
            src=image,
            cameraMatrix=self.__camera_matrix,
            distCoeffs=self.__dist_coeffs
        )
        self.__image_saver.save('undistort', self.current_filename, image)
        return image

    def __get_binary_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        white_mask = cv2.inRange(
            image,
            lowerb=self.__white_lane_hsv_range[0],
            upperb=self.__white_lane_hsv_range[1]
        )

        yellow_mask = cv2.inRange(
            image,
            lowerb=self.__yellow_lane_hsv_range[0],
            upperb=self.__yellow_lane_hsv_range[1]
        )
        image = cv2.bitwise_or(
            white_mask,
            yellow_mask
        )

        image = cv2.Canny(image)


        self.__image_saver.save('binary', self.current_filename, image)

        return image

    def process(self, image):
        source_image = image.copy()

        image = self.__undistort(image)
        image = self.__get_binary_image(image)

        return image


def main():
    save_images = True
    image_saver = ImageSaver('./output_images', save_images)
    calibration_file = './calibration.p'
    if os.path.isfile(calibration_file):
        with open(calibration_file, 'rb') as file:
            retval, camera_matrix, dist_coeffs, rvecs, tvecs = pickle.load(file)
    else:
        calibration_images = glob.glob('./camera_cal/calibration*.jpg')
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate(
            calibration_images=calibration_images,
            chessboard_shape=(9, 6),
            image_saver=image_saver
        )
        with open(calibration_file, 'wb') as file:
            pickle.dump((retval, camera_matrix, dist_coeffs, rvecs, tvecs), file)

    yellow_lower_bound = np.array([int(0.2 * 255), int(0.3 * 255), int(0.10 * 255)], dtype="uint8")
    yellow_upper_bound = np.array([int(0.6 * 255), int(0.8 * 255), int(0.90 * 255)], dtype="uint8")

    white_lower_bound = np.array([int(0.0 * 255), int(0.0 * 255), int(0.75 * 255)], dtype="uint8")
    white_upper_bound = np.array([int(1.0 * 255), int(0.1 * 255), int(1.0 * 255)], dtype="uint8")

    pipeline = Pipeline(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_saver=image_saver,
        yellow_lane_hsv_range=(
            yellow_lower_bound,
            yellow_upper_bound
        ),
        white_lane_hsv_range=(
            white_lower_bound,
            white_upper_bound
        )
    )
    test_images = glob.glob('./test_images/*.jpg')

    for test_image in test_images:
        filename = test_image.split('/')[-1]
        pipeline.current_filename = filename

        image = cv2.imread(test_image)
        image = pipeline.process(image)
        cv2.imwrite('./output_images/' + filename, image)


if __name__ == "__main__":
    main()
