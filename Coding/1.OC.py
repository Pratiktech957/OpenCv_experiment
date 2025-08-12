import cv2
import numpy as np

def main():
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    # Draw a blue rectangle (BGR format)
    cv2.rectangle(img, (100, 100), (400, 400), (255, 0, 0), -1)

    # Draw a red circle
    cv2.circle(img, (256, 256), 50, (0, 0, 255), -1)

    # Draw a green line
    cv2.line(img, (0, 0), (511, 511), (0, 255, 0), 5)

    # Show the image
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# This ensures main() runs when script is executed
if __name__ == "__main__":
    main()
