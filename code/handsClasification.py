import cv2
import numpy as np
image = cv2.imread('code/input_image.jpg')
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
