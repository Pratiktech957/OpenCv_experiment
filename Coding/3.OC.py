import cv2


image = cv2.imread("Coding/pic.png")

if image is not None:
   success =  cv2.imwrite("Coding/pic_copy.png", image)
   print("Image saved successfully:", success)
    
   cv2.imshow("Image", image)
   if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows() 
        
else:
    print("Error: Image not found.")
    