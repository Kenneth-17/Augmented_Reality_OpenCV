import cv2
import numpy as np

def load_camera_parameters(file_path):
    with np.load(file_path) as data:
        cameraMatrix = data['cameraMatrix']
        distCoeffs = data['distCoeffs']
    return cameraMatrix, distCoeffs

# Load the template image (marker)
template = cv2.imread('template_closed.png')
if template is None:
    raise FileNotFoundError("The specified template image did not load.")

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=500)

# Find keypoints and descriptors in the resized template
kp1, des1 = orb.detectAndCompute(template, None)

# Load camera parameters
cameraMatrix, distCoeffs = load_camera_parameters('calibration_output.npz')

# Start video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define 3D points of a cube (assuming the cube is 1x1x1 units in size)
objectPoints = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
], dtype=np.float32)

# Initialize BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detect and compute keypoints and descriptors in the live frame
    kp2, des2 = orb.detectAndCompute(frame, None)

    # Match descriptors
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Map the object points based on the homography
            h, w = template.shape[:2]
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            imagePoints = dst.reshape(-1, 2)

            # Solve for pose using RANSAC
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objectPoints[:4], imagePoints, cameraMatrix, distCoeffs)
            if inliers is not None:
                imgpts, _ = cv2.projectPoints(objectPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
                if not np.any(np.isnan(imgpts)) and not np.any(np.isinf(imgpts)):
                    img = frame.copy()
                    imgpts = np.int32(imgpts).reshape(-1, 2)
                    
                    # Draw the cube
                    cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
                    for i, j in zip(range(4), range(4, 8)):
                        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
                    cv2.drawContours(img, [imgpts[4:8]], -1, (0, 0, 255), 3)
                    
                    # Display the image
                    cv2.imshow('3D Projection on Real-Time Webcam Feed', img)
                else:
                    print("Invalid projection points detected.")
            else:
                print("Pose estimation failed.")
        else:
            print("Homography calculation failed.")

    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
