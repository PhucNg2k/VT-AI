import cv2
import numpy as np

# 1. Define the position
ROBOT_POS = (1000, 200)  # (x, y) coordinate

# 2. Load the image
# Replace 'your_image.jpg' with the path to your image file
try:
    img = cv2.imread(r'../../Public data task 3\Public data\Public data train\rgb\0000.png')
    print(img.shape)
    print(type(img))
    
    if img is None:
        raise FileNotFoundError("Image not found. Please check the file path.")
except FileNotFoundError as e:
    # Create a dummy image if the file is not found (for demonstration)
    print(f"Warning: {e}. Creating a black placeholder image instead.")
    img_width, img_height = 1200, 800
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8) # Black image
    # Ensure the point is visible on the dummy image
    if ROBOT_POS[0] >= img_width or ROBOT_POS[1] >= img_height:
        print("Note: ROBOT_POS is outside the placeholder image size (1200x800).")

# 3. Draw a circle (dot) at the position
center_coordinates = ROBOT_POS
radius = 7
color = (0, 0, 255)    # BGR color: Red
thickness = -1         # -1 means filled circle

cv2.circle(img, center_coordinates, radius, color, thickness)

# Optional: Add text label
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Robot', (ROBOT_POS[0] + 15, ROBOT_POS[1] + 5), font, 1, color, 2, cv2.LINE_AA)

# 4. Display the image
cv2.imshow('Robot Position Visualization', img)
cv2.waitKey(0)      # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()