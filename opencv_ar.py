import cv2
import numpy as np

def initialize_sift_flann():
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return sift, flann

def draw(img, corner, imgpts):
    corner = tuple(int(x) for x in corner.ravel())
    for pt in imgpts:
        pt_ravel = pt.ravel()
        if np.isfinite(pt_ravel).all() and len(pt_ravel) == 2:
            end_pt = tuple(int(x) for x in pt_ravel)
            if 0 <= end_pt[0] < img.shape[1] and 0 <= end_pt[1] < img.shape[0]:
                img = cv2.line(img, corner, end_pt, (255, 0, 0), 5)
            else:
                print(f"Point out of bounds: {end_pt}")
        else:
            print(f"Invalid point skipped: {pt_ravel}")
    return img

def load_camera_parameters(npz_file_path):
    with np.load(npz_file_path) as data:
        camera_matrix = data['cameraMatrix']
        dist_coeffs = data['distCoeffs']
    return camera_matrix, dist_coeffs

def main():
    camera_matrix, dist_coeffs = load_camera_parameters('calibration_data.npz')
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

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

        if des_frame is not None and len(des_frame) > 0:
            matches = flann.knnMatch(des_marker, des_frame, k=2)
            good_matches = [m for m, n in matches if m and n and m.distance < 0.7 * n.distance]

            if len(good_matches) > 10:
                points_marker = np.float32([kp_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                points_frame = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(points_marker, points_frame, cv2.RANSAC, 5.0)
                if H is not None and np.isfinite(H).all():
                    h, w = custom_marker.shape[:2]
                    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, H)
                    frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

                    axis_length = 50.0
                    axes_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]])
                    imgpts, jac = cv2.projectPoints(axes_points, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                    frame = draw(frame, dst[0], imgpts)

                    cv2.imshow('Warped Marker with Axes', frame)
                else:
                    print("Invalid or unstable homography matrix detected.")
            else:
                print("Not enough good matches to compute homography.")
        else:
            cv2.imshow('Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
