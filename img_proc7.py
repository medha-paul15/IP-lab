
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "assignment7_outputs"
OUT.mkdir(exist_ok=True)

def read_image():
    return np.array(Image.open(BASE / "photos" / "cat1.png").convert("L"), dtype=float)

def save_image(name, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(OUT / name)

def normalize(img):
    return (img - img.min()) * 255 / (img.max() - img.min() + 1e-10)

def convolve(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    result = np.zeros_like(img)
    kernel = kernel[::-1, ::-1]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)

    return result

def gaussian_kernel(size=5, sigma=3):
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x, y = i-center, j-center
            kernel[i, j] = np.exp(-(x*x+y*y)/(2*sigma*sigma))

    return kernel / np.sum(kernel)

def sobel(img):
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    gx = convolve(img, kx)
    gy = convolve(img, ky)

    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)

    return gx, gy, magnitude, angle

def non_maximum_suppression(magnitude, angle):
    h, w = magnitude.shape
    result = np.zeros((h, w))
    direction = (angle * 180 / np.pi) % 180

    for i in range(1, h-1):
        for j in range(1, w-1):
            a = direction[i, j]

            if a < 22.5 or a >= 157.5:
                p, q = magnitude[i, j-1], magnitude[i, j+1]
            elif a < 67.5:
                p, q = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif a < 112.5:
                p, q = magnitude[i-1, j], magnitude[i+1, j]
            else:
                p, q = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if magnitude[i, j] >= p and magnitude[i, j] >= q:
                result[i, j] = magnitude[i, j]

    return result

def hysteresis(img):
    high = img.max() * 0.20
    low = img.max() * 0.08

    strong = 255
    weak = 75

    result = np.zeros(img.shape, dtype=np.uint8)
    result[img >= high] = strong
    result[(img >= low) & (img < high)] = weak

    changed = True

    while changed:
        changed = False

        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                if result[i, j] == weak:
                    if np.any(result[i-1:i+2, j-1:j+2] == strong):
                        result[i, j] = strong
                        changed = True

    result[result != strong] = 0
    return result

def canny(img):
    smooth = convolve(img, gaussian_kernel(5, 1))
    gx, gy, magnitude, angle = sobel(smooth)
    thin = non_maximum_suppression(magnitude, angle)
    return hysteresis(thin)

def main():
    img = read_image()

    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    laplacian = convolve(img, laplacian_kernel)
    sharpened = img - laplacian

    gx, gy, sobel_mag, angle = sobel(img)
    canny_edges = canny(img)

    blurred = convolve(img, gaussian_kernel(5, 3))
    mask = img - blurred
    unsharp = img + 1.5 * mask

    save_image("original.png", img)
    save_image("laplacian.png", sharpened)
    save_image("sobel.png", normalize(sobel_mag))
    save_image("canny.png", canny_edges)
    save_image("blurred.png", blurred)
    save_image("unsharp.png", unsharp)

    images = [img, sharpened, normalize(sobel_mag),
              canny_edges, unsharp]

    titles = ["Original", "Laplacian",
              "Sobel", "Canny", "Unsharp Masking"]

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
