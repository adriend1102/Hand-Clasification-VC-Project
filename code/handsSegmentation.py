import cv2
import numpy as np
src = cv2.imread('code/img0.jpg')

# definir elemento estructurante
kernel = np.ones((30,30), np.uint8)
kernel2 = np.ones((2,2), np.uint8)



src = cv2.GaussianBlur(src, (3, 3), 0)
scale = 1
delta = 0
ddepth = cv2.CV_16S
window_name = ('Sobel Demo - Simple Edge Detector')  
    
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    
grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
# grad_y = cv2.Scharr(gray,ddepth,0,1)
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
_, grad = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)  # umbraliza la imagen
cv2.imshow("grad", grad)
grad = cv2.erode(grad, kernel2, iterations=2)
#grad = cv2.dilate(grad, kernel, iterations=1)
cv2.imshow(window_name, grad)
cv2.waitKey(0)
    