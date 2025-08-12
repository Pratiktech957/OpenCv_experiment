import cv2 
img = cv2.imread("Coding/pic.png")
# print(img)

if img is None:
    raise FileNotFoundError("pic.png not found!")


print("shape ",img.shape)
print("DataType",img.dtype)


print("Pixel value at (50, 100):", img[50, 100])  # [B, G, R]
print("Blue channel value:", img[50, 100, 0])     # 0th index = Blue
print("Green channel value:", img[50, 100, 1])    # 1st index = Green
print("Red channel value:", img[50, 100, 2])      # 2nd index = Red

# Modify pixel color
img[50, 100] = [0, 0, 255]  # Pure red pixel

print(img)
cv2.imshow("Modified", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
