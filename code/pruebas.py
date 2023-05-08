import cv2
import numpy as np
image = cv2.imread('imgtest.jpg')
thresh1 = 0
thresh2 = 255
head = image.shape

# definir elemento estructurante
kernel = np.ones((25,25), np.uint8)
kernel2 = np.ones((5,5), np.uint8)


for i in range(head[0]):
    for j in range(head[1]):
        # Read pixel color
        color = image[i, j] / 255.0
        # Find max and min values of color channels
        mx = max(color[0], color[1], color[2])
        mn = min(color[0], color[1], color[2])
        delta = mx - mn
        if delta == 0:
            h=0
        else:
            # Calculate hue value
            if mx == color[0]:
                h = (color[1] - color[2]) / delta
            elif mx == color[1]:
                h = 2 + (color[2] - color[0]) / delta
            else:
                h = 4 + (color[0] - color[1]) / delta

        h = h * 60

        if h < 0:
            h += 360

        if thresh1 <= h <= thresh2:
            # Identify pixel as skin
            image[i, j] = [0, 0, 0]  # set pixel color to black
        else:
            image[i, j] = [255, 255, 255]  # set pixel color to white

# dilatar imagen
image = cv2.dilate(image, kernel, iterations=1)
cv2.imshow('Skin Pixel Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2

image = cv2.imread('input_image.jpg')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Set the lower and upper threshold values for skin color
lower_skin = (0, 20, 70)
upper_skin = (20, 255, 255)

# Create a mask of the pixels that fall within the skin color range
mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

# Apply the mask to the original image
#result = cv2.bitwise_and(image, image, mask=mask)
# Invert the mask
mask_inv = cv2.bitwise_not(mask)

# Apply the inverted mask to the original image
result = cv2.bitwise_and(image, image, mask=mask_inv)


# Show the result
cv2.imshow('Skin Pixel Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Convert the result to grayscale
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the grayscale image
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)




# dilatar imagen
dilation = cv2.dilate(binary, kernel2, iterations=1)
# erosionar imagen
erosion = cv2.erode(dilation, kernel, iterations=1)

erosion = cv2.dilate(erosion, kernel, iterations=1)

# mostrar resultados
cv2.imshow('Original', binary)
cv2.imshow('Erosion', erosion)
#cv2.imshow('Dilatacion', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Show the result
cv2.imshow('Skin Pixel Detection', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()