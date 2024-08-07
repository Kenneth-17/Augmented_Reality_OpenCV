import cv2
import numpy as np

def initialize_sift_flann():
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # FLANN parameters and matcher setup
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return sift, flann

def main():
    # Load the custom marker image
    custom_marker = cv2.imread('template_closed.png')
    if custom_marker is None:
        print("Error loading marker image.")
        return

    # Convert the marker image to grayscale
    gray_marker = cv2.cvtColor(custom_marker, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT and FLANN
    sift, flann = initialize_sift_flann()
    
    # Detect keypoints and descriptors in the marker image
    kp_marker, des_marker = sift.detectAndCompute(gray_marker, None)

    # Start video capture from the webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and compute keypoints and descriptors in the live frame
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

        if des_frame is not None and len(des_frame) > 0:
            # Match descriptors using FLANN
            matches = flann.knnMatch(des_marker, des_frame, k=2)

            # Apply ratio test to find good matches
            good_matches = []
            for m, n in matches:
                if m and n and m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:
                # Extract location of good matches
                points_marker = np.float32([kp_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                points_frame = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography
                H, mask = cv2.findHomography(points_marker, points_frame, cv2.RANSAC, 5.0)
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
