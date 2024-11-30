import cv2
import numpy as np

# Set up camera
cap = cv2.VideoCapture(0)

# Parameters for Shi-Tomasi Corner Detection to find points for Lucas-Kanade
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial points to track using Shi-Tomasi Corner Detection
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask for drawing Lucas-Kanade optical flow tracks
mask = np.zeros_like(old_frame)

while True:
    # Capture a new frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ---- Lucas-Kanade Optical Flow (Sparse) ----
    # Calculate optical flow using Lucas-Kanade for sparse feature points
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points (those successfully tracked)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color=(0, 255, 0), thickness=2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color=(0, 0, 255), thickness=-1)
        img = cv2.add(frame, mask)
        cv2.imshow('Optical Flow - Lucas-Kanade', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        break

cap.release()
cv2.destroyAllWindows()

# _______________Additional example_______________

# Dense Optical Flow using Farneback's method

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback's method
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2,
                                        flags=0)

    # Convert flow to RGB representation
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Dense Optical Flow - Farneback', rgb_flow)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prvs = next

cap.release()
cv2.destroyAllWindows()
