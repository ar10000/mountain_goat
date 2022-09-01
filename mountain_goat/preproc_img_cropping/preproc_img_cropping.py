import os
import cv2
#from PIL import Image
import numpy as np

#change image path to cloud
image_path = "/Users/andrew/Desktop/Screenshot(1071).png"
image = cv2.imread(image_path)
print(np.shape(image))

cropped_image = image[50:1030, 300:1620]
np.shape(cropped_image)
cv2.imshow("cropped", cropped_image)
cv2.imwrite("Cropped Image.jpg", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
