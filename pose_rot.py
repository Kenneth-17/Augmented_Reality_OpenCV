import cv2
import numpy as np

def prepare_template(path, label):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {path} could not be loaded. Check the file path.")
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp, des = sift.detectAndCompute(image, None)
    return (image, kp, des, label)

templates = [
    prepare_template('template_closed.png', 'Gripper Closed'), 
    prepare_template('template_open.png', 'Gripper Open')
]

# Use the actual calibration data
camera_matrix = np.array([[1.12246945e+03, 0.00000000e+00, 4.71182564e+02],
                          [0.00000000e+00, 1.12908270e+03, 3.99513697e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_coeffs = np.array([0.17334803, -0.94422561, 20.00945764, -112.661219])  # Assuming these are the radial and tangential distortion coefficients

sift = cv2.SIFT_create(nfeatures=1000)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def feature_match(img, templates):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None)
    best_img = img  # Default to no detection
    max_matches = 0

    for template, kp1, des1, label in templates:
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if n and m.distance < 0.60 * n.distance]

        if len(good_matches) > max_matches and len(good_matches) > 15:
            max_matches = len(good_matches)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None and len(mask) >= 4:
                _, rotations, translations, _ = cv2.decomposeHomographyMat(M, camera_matrix)
                if rotations:
                    rotation_matrix = rotations[0]  # Take the first solution as an example
                    translation_vector = translations[0]

                    h, w = template.shape
                    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, M)
                    best_img = cv2.polylines(img.copy(), [np.int32(transformed_corners)], True, 255, 3, cv2.LINE_AA)
                    cv2.putText(best_img, label, (int(transformed_corners[0][0][0]), int(transformed_corners[0][0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(best_img, f"Rotation: {np.rad2deg(np.arccos(rotation_matrix[0, 0])):.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    cv2.putText(best_img, f"Distance: {np.linalg.norm(translation_vector):.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return best_img

cap = cv2.VideoCapture(1)  # Update camera index if necessary
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = feature_match(frame, templates)
    cv2.imshow('SIFT Feature Matching with Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
