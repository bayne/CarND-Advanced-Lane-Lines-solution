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
        """
        :param output_directory: The directory to write the sub directories for the output images
        :param enabled: Set to true to enable writing of the images
        """
        self.enabled = enabled
        self.__output_directory = output_directory

    def save(self, sub_directory, filename, image):
        """
        Saves the image to the disk
        
        :param sub_directory: The subdirectory for the image to fall under
        :param filename: The filename of the image
        :param image: The image data
        :return: 
        """
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
            source_points,
            destination_points,
            window_margin,
            window_min,
            region
    ) -> None:
        """
        :param camera_matrix: The distortion correction matrix
        :param dist_coeffs: 
        :param image_saver: an ImageSaver to use to save the images per pipeline component
        :param yellow_lane_hsv_range: an ordered pair of tuples that specify the range of the yellow lane in HSV 
                                      colorspace
        :param white_lane_hsv_range: an ordered pair of tuples that specify the range of the white lane in HSV 
                                     colorspace
        :param ksize: The size of the kernel for the Sobel filter
        :param gradient_x_threshold: The threshold of the gradient in the x direction for the Sobel filter
        :param gradient_y_threshold: The threshold of the gradient in the y direction for the Sobel filter
        :param gradient_magnitude_threshold: The threshold of the magnitude for the gradient in the Sobel filter
        :param gradient_direction_threshold: The threshold of the direction for the gradient in the Sobel filter
        :param source_points: The source points to use in the perspective transform this should be outlining the length 
                              of a dash in a lane line and the width of a lane (10ft by 12ft)
        :param destination_points: The destination points of the perspective transform
        :param window_margin: The margin for the sliding window for the lane finding
        :param window_min: The minimum step for the sliding window
        :param region: The region of interest to mask for find the lane lines
        """
        self.__destination_points = destination_points
        self.__region = region
        self.__window_min = window_min
        self.__window_margin = window_margin
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

    def __undistort(self, image):
        """
        Undistorts the image
        :param image: The image data
        :return: An undistorted image
        """
        image = cv2.undistort(
            src=image,
            cameraMatrix=self.__camera_matrix,
            distCoeffs=self.__dist_coeffs
        )
        self.__image_saver.save('undistort', self.current_filename, image)
        return image

    def __color_threshold(self, image):
        """
        Removes pixels that are not within the color ranges
        :param image: 
        :return: 
        """
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
        """
        Using the Sobel filter, find the edge pixels for the lane lines
        :param image: 
        :return: 
        """
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

    def __perspective_transform(self, image, source_image):
        """
        Transforms the image to a birds-eye view
        :param image: 
        :param source_image: 
        :return: 
        """
        image_shape = (image.shape[1], image.shape[0])

        destination_points = self.__destination_points

        transformation_matrix = cv2.getPerspectiveTransform(self.__source_points, destination_points)
        reverse_transformation_matrix = cv2.getPerspectiveTransform(destination_points, self.__source_points)

        def warp(matrix, image, image_shape):
            return cv2.warpPerspective(src=image, M=matrix, dsize=image_shape, flags=cv2.INTER_LINEAR)

        image = warp(transformation_matrix, image, image_shape)

        perspective_image = warp(transformation_matrix, source_image, image_shape)
        self.__image_saver.save('perspective', self.current_filename, perspective_image)
        return image, reverse_transformation_matrix

    def __lane_pixels(self, image):
        """
        Finds the pixels associated to the lane using a sliding window
        :param image: 
        :return: 
        """

        binary_warped = image

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        half = int(binary_warped.shape[0] / 2)
        histogram = np.sum(binary_warped[half:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = self.__window_margin
        # Set minimum number of pixels found to recenter window
        minpix = self.__window_min
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        self.__image_saver.save(filename=self.current_filename, sub_directory='lane_pixels', image=out_img)

        return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty

    def __annotate_lane(self, source_image, warped_image, reverse_matrix, left_fitx, right_fitx, ploty):
        """
        Draws the detected region of the lane lines
        :param source_image: 
        :param warped_image: 
        :param reverse_matrix: Used for undistorting the image
        :param left_fitx: 
        :param right_fitx: 
        :param ploty: 
        :return: 
        """
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = warp_zero

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, reverse_matrix, (source_image.shape[1], source_image.shape[0]))
        # Combine the result with the original image
        return cv2.addWeighted(source_image, 1, newwarp, 0.3, 0)

    def __lane_stats(self, image_shape, left_fit, right_fit, ploty, leftx, rightx, lefty, righty):
        """
        Generates stats about the position and curvature of the lane
        :param image_shape: 
        :param left_fit: 
        :param right_fit: 
        :param ploty: 
        :param leftx: 
        :param rightx: 
        :param lefty: 
        :param righty: 
        :return: 
        """
        y_eval = image_shape[1]

        ym_per_pix = 12.0 / (self.__destination_points[3][1] - self.__destination_points[0][1])
        xm_per_pix = 10.0 / (self.__destination_points[1][0] - self.__destination_points[0][0])

        left = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        right = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

        # feetPerPixel = 12 / (rightx - leftx)

        lane_midpoint_px = (right + left) / 2
        camera_midpoint_px = image_shape[0] / 2

        offset_from_center = np.abs(lane_midpoint_px - camera_midpoint_px) * xm_per_pix

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

        return offset_from_center, np.min([left_curverad, right_curverad])

    def __display_numbers(self, image, offset_from_center, curverad):
        cv2.putText(image, 'Offset: {:.1f} Curve radius: {:.1f}'.format(offset_from_center, curverad), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    def __region_of_interest(self, img, source_image):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """

        points = np.array([
            self.__region[3],
            self.__region[0],
            self.__region[1],
            self.__region[2],
        ])

        vertices = np.int32([points])

        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)

        annotated_image = source_image.copy()
        cv2.fillPoly(annotated_image, vertices, (255, 255, 255))
        # annotated_image= cv2.addWeighted(annotated_image, 1, mask, 0.3, 0)

        self.__image_saver.save(filename=self.current_filename, sub_directory='region', image=annotated_image)

        return masked_image

    def process(self, image):
        """
        Find the lane lines in the provided image
        :param image: 
        :return: 
        """
        source_image = image.copy()

        image = self.__region_of_interest(image, source_image)

        image = self.__undistort(image)

        image, reverse = self.__perspective_transform(image, source_image)

        image = self.__color_threshold(image)
        image = self.__edge_detection(image)

        image, left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx,lefty, righty = self.__lane_pixels(image)
        image = self.__annotate_lane(source_image=source_image, warped_image=image, reverse_matrix=reverse,
                                     left_fitx=left_fitx, right_fitx=right_fitx, ploty=ploty)
        offset_from_center, curverad = self.__lane_stats(image_shape=(source_image.shape[1], source_image.shape[0]),
                          left_fit=left_fit, right_fit=right_fit, ploty=ploty, leftx=leftx, rightx=rightx, lefty=lefty, righty=righty)
        self.__display_numbers(image, offset_from_center=offset_from_center, curverad=curverad)

        return image


def get_pipeline(image_saver):
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

    white_lower_bound = np.array([int(0.0 * 255), int(0.0 * 255), int(0.80 * 255)], dtype="uint8")
    white_upper_bound = np.array([int(1.0 * 255), int(0.10 * 255), int(1.0 * 255)], dtype="uint8")

    ratio = (10, 12)
    scale = 10
    offset = (600, 300)

    source_points = np.array([
        [540, 488],
        [750, 488],
        [777, 508],
        [507, 508]
    ], dtype=np.float32)

    destination_points = np.array([
        [offset[0], offset[1]],
        [offset[0] + ratio[0] * scale, offset[1]],
        [offset[0] + ratio[0] * scale, offset[1] + ratio[1] * scale],
        [offset[0], offset[1] + ratio[1] * scale]
    ], dtype=np.float32)

    return Pipeline(
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
        ksize=7,
        gradient_x_threshold=(0, 30),
        gradient_y_threshold=(20, 90),
        gradient_magnitude_threshold=(0, 10),
        gradient_direction_threshold=(0, np.pi / 4),
        source_points=source_points,
        destination_points=destination_points,
        window_margin=20,
        window_min=5,
        region=np.array([
            [580, 435],
            [700, 435],
            [1100, 660],
            [190, 660]
        ])
    )


def process_test_images():
    save_images = True
    image_saver = ImageSaver('./output_images', save_images)

    pipeline = get_pipeline(image_saver)

    test_images = glob.glob('./test_images/*.jpg')

    for test_image in test_images:
        filename = test_image.split('/')[-1]
        pipeline.current_filename = filename

        image = cv2.imread(test_image)
        image = pipeline.process(image)
        cv2.imwrite('./output_images/' + filename, image)


from moviepy.editor import VideoFileClip


def process_video():
    image_saver = ImageSaver('./output_images', False)

    pipeline = get_pipeline(image_saver)

    clip = VideoFileClip(filename="./project_video.mp4")

    pipeline.current_filename = "0_frame.png"

    def process(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = pipeline.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    clip = clip.fl_image(process)
    clip.write_videofile(filename="./project_video_output.mp4", audio=False)

def process_challenge_video():
    image_saver = ImageSaver('./output_images', False)

    pipeline = get_pipeline(image_saver)

    clip = VideoFileClip(filename="./challenge_video.mp4")

    pipeline.current_filename = "0_frame.png"

    def process(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = pipeline.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    clip = clip.fl_image(process)
    clip.write_videofile(filename="./challenge_video_output.mp4", audio=False)


def main():
    # process_test_images()
    process_challenge_video()
    # process_video()


if __name__ == "__main__":
    main()
