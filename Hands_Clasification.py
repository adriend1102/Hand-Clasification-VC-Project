import cv2
image = cv2.imread('input_image.jpg')
thresh1 = 50
thresh2 = 255
head = image.shape

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

        # Check if pixel is within threshold
        if thresh1 <= h <= thresh2:
            # Identify pixel as skin
            image[i, j] = [0, 255, 0]  # set pixel color to green
cv2.imshow('Skin Pixel Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()