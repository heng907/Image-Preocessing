import numpy as np
import cv2

# define Laplacian kernel
# 4-neighbor Laplacian
kernel_1 = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]], dtype = np.float32)
# 8-neighbor Laplacian
kernel_2 = np.array([[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]], dtype = np.float32)

# read image
img = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
# normalized
img = img / 255
# convolution
def conv (image, kernel):

    h, w = image.shape
    kh, kw = kernel.shape

    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode = 'constant')

    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    return result

# compute convolution
laplacian = conv(img, kernel_2)

# sharpen -> g(x, y) = f(x, y) - c * laplacian

c = 1.0
sharpened = img - c * laplacian
sharpened = np.clip(sharpened, 0, 1)

# show result
cv2.imshow('sharpened', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
