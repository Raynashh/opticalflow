import cv2
import numpy as np

# Initialize video capture (0=webcam / replace with video path)
cap = cv2.VideoCapture("placeholder.MOV")  
# Read first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Couldn't read video")
    exit()

# Convert to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Create HSV image for visualization (Hue=Saturation=Value)
hsv_mask = np.zeros_like(prev_frame)
hsv_mask[..., 1] = 255  # Set saturation to maximum

while True:
    # Read next frame
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calc Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, 
        None,  # No flow initialization
        pyr_scale=0.5,  # Image scale (<1 for pyramids)
        levels=3,  # Number of pyramid layers
        winsize=15,  # Averaging window size
        iterations=3,  # Iterations per level
        poly_n=5,  # Pixl neighborhd size
        poly_sigma=1.2,  # Gaussian sigma
        flags=0
    )
    
    # Convert flow to polar coords (magnitude / angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Update HSV visualization
    hsv_mask[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
    
    # Convert HSV to RGB for display
    flow_rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    
    # Show original + optical flow
    combined = np.hstack((frame, flow_rgb))
    cv2.imshow("Optical Flow", combined)
    
    # Exit on q
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    # Update previous frame
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
