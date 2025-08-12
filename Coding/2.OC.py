import cv2 


def main():

 image = cv2.imread("Coding/pic.png")

 if image is None:
    print("Error: Image not found.")
 else:
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()  