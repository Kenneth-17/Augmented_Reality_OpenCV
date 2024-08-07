import cv2
import numpy as np

def draw(img, corner, imgpts):
    if imgpts is None:
        print("No valid points available for drawing.")
        return img
    try:
        corner = tuple(int(x) for x in corner.ravel())
        for pt in imgpts:
            end_pt = tuple(int(x) for x in pt.ravel())
            if all(0 <= x < img.shape[i] for i, x in enumerate(end_pt)):
                img = cv2.line(img, corner, end_pt, (255, 0, 0), 5)  # Draw in red
            else:
                print(f"Point out of bounds: {end_pt}")
    except Exception as e:
        print(f"Error drawing line: {e}")
    return img

def main():
    camera_matrix, dist_coeffs = load_camera_parameters('calibration_output.npz')
    # Initialization and image loading omitted for brevity
    while True:
        # Image capture and preprocessing omitted for brevity
        matches = flann.knnMatch(des_marker, des_frame, k=2)
        # Match filtering omitted for brevity

        if len(good_matches) > 10:
            points_marker = np.float32([kp_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            points_frame = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(points_marker, points_frame, cv2.RANSAC, 5.0)

            if H is not None and np.isfinite(H).all():
                # Ensure H is valid before proceeding
                print("Homography Matrix:", H)
                # Further processing and drawing omitted for brevity
            else:
                print("Invalid or unstable homography matrix detected.")
        # Additional processing omitted for brevity

if __name__ == "__main__":
    main()
