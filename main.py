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
            white_lane_hsv_range,
            ksize,
            gradient_x_threshold,
            gradient_y_threshold,
            gradient_magnitude_threshold,
            gradient_direction_threshold,
            source_points
    ) -> None:
        self.__source_points = source_points
        self.__ksize = ksize
        self.__gradient_x_threshold = gradient_x_threshold
        self.__gradient_y_threshold = gradient_y_threshold
        self.__gradient_magnitude_threshold = gradient_magnitude_threshold
        self.__gradient_direction_threshold = gradient_direction_threshold
        self.__white_lane_hsv_range = white_lane_hsv_range
        self.__yellow_lane_hsv_range = yellow_lane_hsv_range
        self.__image_saver = image_saver
        self.__camera_matrix = camera_matrix
        self.__dist_coeffs = dist_coeffs
        self.current_filename = None

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

    def __color_threshold(self, image):
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
        self.__image_saver.save('color', self.current_filename, image)

        return image

    def __edge_detection(self, image):

        # Define a function that takes an image, gradient orientation,
        # and threshold min / max values.
        def abs_sobel_thresh(img, orient='x', thresh=None, sobel_kernel=None):
            thresh_min = thresh[0]
            thresh_max = thresh[1]
            # Convert to grayscale
            gray = img
            # Apply x or y gradient with the OpenCV Sobel() function
            # and take the absolute value
            if orient == 'x':
                abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
            elif orient == 'y':
                abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
            else:
                raise Exception('Invalid `orient`')
            # Rescale back to 8 bit integer
            scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
            # Create a copy and apply the threshold
            binary_output = np.zeros_like(scaled_sobel)
            # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
            binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

            # Return the result
            return binary_output

        # Define a function to return the magnitude of the gradient
        # for a given sobel kernel size and threshold values
        def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
            # Convert to grayscale
            gray = img
            # Take both Sobel x and y gradients
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            # Calculate the gradient magnitude
            gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
            # Rescale to 8 bit
            scale_factor = np.max(gradmag) / 255
            gradmag = (gradmag / scale_factor).astype(np.uint8)
            # Create a binary image of ones where threshold is met, zeros otherwise
            binary_output = np.zeros_like(gradmag)
            binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

            # Return the binary image
            return binary_output

        def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
            # Grayscale
            # Calculate the x and y gradients
            gray = image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            # Take the absolute value of the gradient direction,
            # apply a threshold, and create a binary image result
            absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
            binary_output = np.zeros_like(absgraddir)
            binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

            # Return the binary image
            return binary_output

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=self.__ksize, thresh=self.__gradient_x_threshold)
        grady = abs_sobel_thresh(image, orient='y', sobel_kernel=self.__ksize, thresh=self.__gradient_y_threshold)
        mag_binary = mag_thresh(image, sobel_kernel=self.__ksize, mag_thresh=self.__gradient_magnitude_threshold)
        dir_binary = dir_threshold(image, sobel_kernel=self.__ksize, thresh=self.__gradient_direction_threshold)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 0) & (grady == 0)) | ((mag_binary == 0) & (dir_binary == 0))] = 255

        self.__image_saver.save('binary', self.current_filename, combined)

        return combined

    def __perspective_transform(self, image):
        image_shape = (image.shape[1], image.shape[0])
        destination_points = np.float32([
            [image_shape[0]/4, 0],
            [3*image_shape[0]/4, 0],
            [3*image_shape[0]/4, image_shape[1]],
            [image_shape[0]/4, image_shape[1]]
        ])
        transformation_matrix = cv2.getPerspectiveTransform(self.__source_points, destination_points)
        reverse_transformation_matrix = cv2.getPerspectiveTransform(destination_points, self.__source_points)

        image = cv2.warpPerspective(src=image, M=transformation_matrix, dsize=image_shape)

        self.__image_saver.save('perspective', self.current_filename, image)
        return image, reverse_transformation_matrix

    def process(self, image):
        source_image = image.copy()

        image = self.__undistort(image)
        image = self.__color_threshold(image)
        image = self.__edge_detection(image)
        image, reverse = self.__perspective_transform(image)

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
        ),
        ksize=3,
        gradient_x_threshold=(0, 50),
        gradient_y_threshold=(0, 50),
        gradient_magnitude_threshold=(0, 50),
        gradient_direction_threshold=(0, np.pi/4),
        source_points=np.float32([
            [618, 440],
            [703, 440],
            [1120, 700],
            [250, 700]
        ])
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
