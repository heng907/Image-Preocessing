import numpy as np
import cv2

img = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
# normalized
img = img / 255
# Step1. Get the Spectrum using Fourier Transform. (ex. np.fft.fft2)
spectrum = np.fft.fft2(img)
F = np.fft.fftshift(spectrum)

# Step2. Apply filter on Spectrum.
# H(u,v) = -4*pi**2[(u-P/2)**2+(v-Q/2)**2]
P, Q = F.shape
H = np.zeros((P, Q), dtype=np.float32)
for u in range(P):
    for v in range(Q):
        H[u, v] = -4 * np.pi * np.pi*((u - P/2)**2 + (v-Q/2)**2)

# Step3. Convert the new Spectrum to spatial domain using inverse Fourier Transform.
# laplacian = F^-1[H(u, v)F(u, v)]
laplacian = H * F
laplacian = np.fft.ifftshift(laplacian)
laplacian = np.real(np.fft.ifft2(laplacian))

old_range = np.max(laplacian) - np.min(laplacian)
new_range = 1 - (-1)
scaled_lap = (((laplacian - np.min(laplacian)) * new_range) / old_range) + (-1)

c = 1.0
sharpened = img - c * scaled_lap
sharpened = np.clip(sharpened, 0, 1)

# show result
cv2.imshow('sharpened', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()

