import cv2
import numpy as np

import matplotlib.pyplot as plt
image = cv2.imread('code/img0.jpg')
thresh1 = 0
thresh2 = 255
head = image.shape

# definir elemento estructurante
kernel = np.ones((30,30), np.uint8)
kernel2 = np.ones((3,3), np.uint8)


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

image = cv2.imread('code/img0.jpg')

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

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
H = np.array(hsv_img[:, :, 0])
S = np.array(hsv_img[:, :, 1])
V = np.array(hsv_img[:, :, 2])

max_val = np.max(hsv_img, axis=2)
min_val = np.min(hsv_img, axis=2)
delta = np.subtract(max_val, min_val)
h = np.zeros_like(H)
# Condiciones para la selección de max
red_idx = (max_val == H) & (delta != 0)
green_idx = (max_val == S) & (delta != 0) & ~red_idx
blue_idx = (max_val == V) & (delta != 0) & ~(red_idx | green_idx)

# Cálculo de h en función de las condiciones
h[red_idx] = ((S[red_idx] - V[red_idx]) / delta[red_idx]) % 6
h[green_idx] = ((V[green_idx] - S[green_idx]) / delta[green_idx]) + 2
h[blue_idx] = ((S[blue_idx] - H[blue_idx]) / delta[blue_idx]) + 4
h *= 60
h[h < 0] += 360
skin_mask = np.zeros_like(H, dtype=np.uint8)
skin_mask[(h >= thresh1) & (h <= thresh2)] = 255
masked_img = cv2.bitwise_and(image, image, mask=skin_mask)
cv2.imshow('Masked Image', masked_img)
cv2.waitKey(0)


"""
# Define display_contour variable
display_contour = True

# Iterate through each pixel in the image
for i in range(1, head[0]):
    for j in range(1, head[1]):
        # Read current pixel and previous pixel
        current_pixel = image[i, j]
        previous_pixel = image[i-1, j-1]
        
        # Check if current pixel is the same color as previous pixel
        if (current_pixel == previous_pixel).all():
            previous_pixel = current_pixel
        # If not, check if current pixel is white
        elif (current_pixel == [255, 255, 255]).all():
            # Mark previous pixel as border pixel
            image[i-1, j-1] = [0, 0, 0]
        # Otherwise, mark current pixel as border pixel
        else:
            # Mark current pixel as border pixel
            image[i, j] = [0, 0, 0]
        
        # If display_contour is enabled, change border pixel color to yellow
        if display_contour == '1' and (image[i, j] == [0, 0, 0]).all():
            image[i, j] = [0, 255, 255]

# Show output image
cv2.imshow('output_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
image = cv2.imread('code/img.jpg')

#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # umbraliza la imagen
# remove noise
image = cv2.GaussianBlur(image,(3,3),0)
#_, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)  # umbraliza la imagen
cv2.imshow('Masked Image', image)
cv2.waitKey(0)
#_, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)  # umbraliza la imagen
sobel = cv2.Sobel(image,cv2.CV_64F,1,1,ksize=5)

sobel_abs = np.absolute(sobel)  # toma el valor absoluto de la imagen
sobel_uint = np.uint8(sobel_abs)  # convierte la imagen a uint8
_, sobel_uint = cv2.threshold(sobel_uint, 60, 255, cv2.THRESH_BINARY)  # umbraliza la imagen
#sobel_uint = cv2.erode(sobel_uint, kernel2, iterations=1)
cv2.imshow('Masked Image', sobel_uint)
cv2.waitKey(0)

"""
# Apply gray scale
gray_img = np.round(0.299 * img[:, :, 0] +
                    0.587 * img[:, :, 1] +
                    0.114 * img[:, :, 2]).astype(np.uint8)
# Sobel Operator
h, w = gray_img.shape
# define filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1
# define images with 0s
newhorizontalImage = np.zeros((h, w))
newverticalImage = np.zeros((h, w))
newgradientImage = np.zeros((h, w))
# offset by 1
for i in range(1, h - 1):
    for j in range(1, w - 1):
        horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) + \
                         (horizontal[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal[2, 2] * gray_img[i + 1, j + 1])
        newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)
        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])
        newverticalImage[i - 1, j - 1] = abs(verticalGrad)
# Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        newgradientImage[i - 1, j - 1] = mag
plt.figure()
plt.title('Butterfly')
plt.imsave('Butterfly.jpg', newgradientImage, cmap='gray', format='jpg')
plt.imshow(newgradientImage, cmap='gray')
plt.show()"""