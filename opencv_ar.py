import cv2
import numpy as np

def initialize_sift_flann():
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return sift, flann

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def load_calibration_data(filename):
    with np.load(filename) as data:
        camera_matrix = data['cameraMatrix']
        dist_coeffs = data['distCoeffs']
    return camera_matrix, dist_coeffs

def main():
    # Load calibration data
    camera_matrix, dist_coeffs = load_calibration_data('calibration_output.npz')

    # Load the custom marker image
    custom_marker = cv2.imread('template_closed.png')
    if custom_marker is None:
        print("Error loading marker image.")
        return

    gray_marker = cv2.cvtColor(custom_marker, cv2.COLOR_BGR2GRAY)
    sift, flann = initialize_sift_flann()
    kp_marker, des_marker = sift.detectAndCompute(gray_marker, None)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Optional: Apply undistortion to the frame
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

        if des_frame is not None and len(des_frame) > 0:
            matches = flann.knnMatch(des_marker, des_frame, k=2)
            good_matches = [m for m, n in matches if m and n and m.distance < 0.7 * n.distance]

            if len(good_matches) > 10:
                points_marker = np.float32([kp_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                points_frame = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(points_marker, points_frame, cv2.RANSAC, 5.0)
                if H is not None:
                    # Warp the custom marker onto the frame using the homography
                    h, w = custom_marker.shape[:2]
                    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, H)
                    frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
                    transformed_marker = cv2.warpPerspective(custom_marker, H, (frame.shape[1], frame.shape[0]))
                    mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillConvexPoly(mask, np.int32(dst), (255,)*frame.shape[2], cv2.LINE_AA)
                    inv_mask = cv2.bitwise_not(mask)
                    frame = cv2.bitwise_and(frame, inv_mask)
                    transformed_marker = cv2.bitwise_and(transformed_marker, mask)
                    frame = cv2.add(frame, transformed_marker)

                    # Show the resulting frame with the warped custom marker
                    cv2.imshow('Warped Marker', frame)
                else:
                    cv2.imshow('Matches', frame)
            else:
                cv2.imshow('Matches', frame)
        else:
            cv2.imshow('Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
