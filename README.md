# Advanced Lane Finding

The goal of this project (from the Udacity Self-driving Car nanodegree):

> In this project, your goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car.

![giphy](https://cloud.githubusercontent.com/assets/712014/25603402/393c0aae-2eb0-11e7-944c-9f669d13eca1.gif)

I used a combination of different computer vision techniques (camera calibration, region of interest, perspective transform) to create a software pipeline to process images and detect traffic lanes. The technologies used to accomplish this:
- Python 3.5
- OpenCV 3

The pipeline consisted of these components:

- Distortion correction
- Region of Interest
- Perspective transform
- Lane pixel detection
- Lane detection
- Curvature inference

## Camera Calibration

Cameras typically have some level of distortion in the images they take. The distortion can cause the image to appear to be warped in some areas. Since we will be using the images to attempt to infer the dimensions of the pictured objects, we need to make sure that the distortion is corrected.

### Calibration Images

![image](https://cloud.githubusercontent.com/assets/712014/25603850/75317ed2-2eb4-11e7-8dfb-9eb718907fc1.png)

Calibration images are a set of images of various calibration objects that have known attributes. By determining the transformation required to go from the known attributes to the actual attributes displayed in the image, we are able to generate a function that can correct for distortion.

Performing the calibration is relatively straight forward (assuming you have multiple calibration images and are using a chessboard diagram):

1. For each of the calibration images find all the corners in the image with [`cv2.findChessboardCorners(image, patternSize[, corners[, flags]])`](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findchessboardcorners#cv2.findChessboardCorners)
2. Generate the transformation matrix for distortion correction using [`cv2.calibrateCamera(objectPoints, imagePoints, imageSize[, cameraMatrix[, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]]]) → retval, cameraMatrix, distCoeffs, rvecs, tvecs`](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findchessboardcorners#cv2.calibrateCamera)

My implementation for this project can be found here:

https://github.com/bayne/CarND-Advanced-Lane-Lines-solution/blob/master/main.py#L9

### Saving the calibration

Since the distortion is a property of camera we only need to calculate the distortion correction matrix once. During my implementation I was going to be running the pipeline many times so to save time I saved the distortion matrix to a pickle file and reloaded it from disk instead of recalculating it.

## Pipeline

The pipeline consisted of [12 tunable parameters](https://github.com/bayne/CarND-Advanced-Lane-Lines-solution/blob/master/main.py#L76) that were used to configure how each step ran:

- Region of interest (`region`)
- Perspective transform (`source_points`, `destination_points`)
- Lane pixel detection
    - Color threshold (`yellow_lane_hsv_range`, `white_lane_hsv_range`)
    - Edge detection (`gradient_x_threshold`, `gradient_y_threshold`, `gradient_magnitude_threshold`, `gradient_direction_threshold`, `ksize`)
- Lane detection (`window_margin`, `window_min`)

### Region of Interest

![straight_lines1](https://cloud.githubusercontent.com/assets/712014/25604949/651da6e4-2ebc-11e7-85d3-b742d4827c10.jpg)

I removed the parts of the image that do not contain lane lines by masking out parts of the image that aren't in the specified `region`.

https://github.com/bayne/CarND-Advanced-Lane-Lines-solution/blob/master/main.py#L439

### Distortion Correction

![straight_lines1](https://cloud.githubusercontent.com/assets/712014/25604974/928c8c3a-2ebc-11e7-9efd-41c26ffccb09.jpg)

Using the pre-calculated distortion correction matrix, the next step is undistort the image:

https://github.com/bayne/CarND-Advanced-Lane-Lines-solution/blob/master/main.py#L131

### Perspective Transform

![straight_lines1](https://cloud.githubusercontent.com/assets/712014/25604983/a88a9270-2ebc-11e7-8dfc-aad03c4f0c1c.jpg)

The image is transformed to a bird's eye view to help accentuate curvature in the road:

https://github.com/bayne/CarND-Advanced-Lane-Lines-solution/blob/master/main.py#L251

### Lane pixel detection

Detecting the lane pixels is done by reducing the image to a binary image of the pixels that belong to lane lines.

#### Color Threshold

![straight_lines1](https://cloud.githubusercontent.com/assets/712014/25605063/1f50ef80-2ebd-11e7-9fcb-c8460f7bb9f1.jpg)

Color thresholding is removing the colors not specified by a given range:

https://github.com/bayne/CarND-Advanced-Lane-Lines-solution/blob/master/main.py#L145

#### Edge Detection 

![straight_lines1](https://cloud.githubusercontent.com/assets/712014/25605108/63d17292-2ebd-11e7-8351-394392d7e207.jpg)

By tuning a Sobel filter to focus on characteristics found in lane lines, I was able to reduce the amount of noise unrelated to lane lines.

https://github.com/bayne/CarND-Advanced-Lane-Lines-solution/blob/master/main.py#L172

### Lane Detection

![straight_lines1](https://cloud.githubusercontent.com/assets/712014/25605130/999896a8-2ebd-11e7-8313-d90c4470bf7b.jpg)

Lanes a detected by using a sliding window that search for pixels that belong to the lane based on the pixels that were detect previously as being part of the lane:

https://github.com/bayne/CarND-Advanced-Lane-Lines-solution/blob/master/main.py#L274

### Position & Curvature

![straight_lines1](https://cloud.githubusercontent.com/assets/712014/25605343/184cc05e-2ebf-11e7-8e47-e07d083afb19.jpg)

I was able to calculate the curvature and position of the car in respect to the lane lines from the image by carefully choosing the `source_points` and `destination_points` used in the perspective transformation step. By using the knowledge that the width of a lane is 12 feet and the length of a lane line is 10 feet, I am able to create a pixel to feet conversion function.

The position of the car in respect to the center of the lane is calculated by finding the offset of the middle of the lane with the middle of the image.

The curvature of the lane is done by using `cv2.fitPoly` which will find a best fit polynomial to the provided points.

https://github.com/bayne/CarND-Advanced-Lane-Lines-solution/blob/master/main.py#L393

### Problems & Improvements

* The region of interest significantly impacts the robustness of the pipeline since it must be tuned for the video feed.
* The color thresholding is also tuned for the particular conditions of the video
* Significant inclines or declines on the road would break the assumption that the birds eye view is on a flat plane.
* Markers on the road that appear lane-line-like (spilled paint) will completely throw off the lane detection.
* Using the information provided by previous frames would increase the smoothness and make it more robust in sudden changes between frames.
