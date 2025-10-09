import cv2

# List to store the clicked coordinates
clicked_points = []

# Define the mouse callback function
def get_pixel_coords(event, x, y, flags, param):
    # Check for a left mouse click down event
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinates to the console
        print(f"Clicked at: X={x}, Y={y}")
        # Store the coordinates
        clicked_points.append((x, y))

        # Optionally, draw a circle at the clicked point for visual confirmation
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Image Window', img)

# Load the image
img = cv2.imread(r'../Public data/Public data train/rgb/0000.png') # Replace 'your_image_file.png' with your file path

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image file.")
else:
    # Create a window to display the image
    cv2.namedWindow('Image Window')

    # Bind the callback function to the window
    cv2.setMouseCallback('Image Window', get_pixel_coords)

    # Display the image and wait for a key press (or until the user closes the window)
    cv2.imshow('Image Window', img)
    cv2.waitKey(0) # 0 means wait indefinitely for a key press

    # Clean up all windows
    cv2.destroyAllWindows()

    print("All clicked points:", clicked_points)