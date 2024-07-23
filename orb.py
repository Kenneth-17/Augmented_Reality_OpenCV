import cv2
import numpy as np

# Function to load a template image, compute its keypoints and descriptors using ORB
def prepare_template(path, label):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {path} could not be loaded. Check the file path.")
    orb = cv2.ORB_create(nfeatures=1500)  # You can adjust the number of features
    kp, des = orb.detectAndCompute(image, None)
    return (image, kp, des, label)

# Load and prepare templates using ORB
templates = [
    prepare_template('template_1.jpg', 'Image 1'),  
    prepare_template('template_2.jpg', 'Image 2')  
]

# FLANN parameters and matcher setup for ORB
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,  # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1)  # 2
search_params = dict(checks=50)  # Increase this for higher accuracy

flann = cv2.FlannBasedMatcher(index_params, search_params)

def feature_match(img, templates):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best_img = img  # Default to no detection
    max_matches = 0

    orb = cv2.ORB_create(nfeatures=1500)
    kp2, des2 = orb.detectAndCompute(gray, None)
    for template, kp1, des1, label in templates:
        if des1 is None or des2 is None:
            continue
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Update best match if current one is better
        if len(good_matches) > max_matches and len(good_matches) > 10:  # Higher threshold for matches
            max_matches = len(good_matches)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = template.shape
                corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, M)
                best_img = cv2.polylines(img.copy(), [np.int32(transformed_corners)], True, 255, 3, cv2.LINE_AA)
                # Draw the label on the bounding box
                cv2.putText(best_img, label, (int(transformed_corners[0][0][0]), int(transformed_corners[0][0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return best_img

# Start webcam
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = feature_match(frame, templates)
    cv2.imshow('ORB Feature Matching for Multiple Images', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
