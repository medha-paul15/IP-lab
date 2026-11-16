
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "assignment10_outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(10)

def read_image():
    return np.array(
        Image.open(BASE / "photos" / "cat1.png").convert("L"),
        dtype=float
    ) / 255.0

def save_image(name, img):
    img = np.clip(img, 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(OUT / name)

def median_filter(img, size=3):
    radius = size // 2
    padded = np.pad(img, radius, mode="reflect")
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+size, j:j+size]
            result[i, j] = np.median(window)

    return result

def adaptive_median_filter(img, initial_size, maximum_size):
    if (initial_size < 3 or maximum_size < initial_size or
        initial_size % 2 == 0 or maximum_size % 2 == 0):
        raise ValueError("Window sizes must be odd and Smax >= initial size.")

    max_radius = maximum_size // 2
    padded = np.pad(img, max_radius, mode="reflect")
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            size = initial_size
            pixel = img[i, j]
            output = pixel

            while size <= maximum_size:
                radius = size // 2

                window = padded[
                    i+max_radius-radius:i+max_radius+radius+1,
                    j+max_radius-radius:j+max_radius+radius+1
                ]

                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)

                a1 = z_med - z_min
                a2 = z_med - z_max

                if a1 > 0 and a2 < 0:
                    b1 = pixel - z_min
                    b2 = pixel - z_max

                    if b1 > 0 and b2 < 0:
                        output = pixel
                    else:
                        output = z_med

                    break

                size += 2

            if size > maximum_size:
                output = z_med

            result[i, j] = output

    return result

def psnr(original, processed):
    mse = np.mean((original - processed)**2)

    if mse == 0:
        return float("inf")

    return 10 * np.log10(1 / mse)

def otsu_threshold(img):
    pixels = (img * 255).astype(np.uint8)

    histogram = np.zeros(256, dtype=int)

    for value in pixels.ravel():
        histogram[value] += 1

    total = pixels.size
    sum_total = 0

    for i in range(256):
        sum_total += i * histogram[i]

    weight_background = 0
    sum_background = 0
    maximum_variance = 0
    threshold = 0

    for t in range(256):
        weight_background += histogram[t]

        if weight_background == 0:
            continue

        weight_foreground = total - weight_background

        if weight_foreground == 0:
            break

        sum_background += t * histogram[t]

        mean_background = sum_background / weight_background

        mean_foreground = (
            sum_total - sum_background
        ) / weight_foreground

        variance = (
            weight_background * weight_foreground *
            (mean_background - mean_foreground)**2
        )

        if variance > maximum_variance:
            maximum_variance = variance
            threshold = t

    return threshold / 255.0

def main():
    img = read_image()

    initial_size = int(input("Enter initial window size: "))
    maximum_size = int(input("Enter maximum window size Smax: "))

    noise = RNG.normal(0, 0.08, img.shape)
    noisy = np.clip(img + noise, 0, 1)

    ordinary = median_filter(noisy, initial_size)

    adaptive = adaptive_median_filter(
        noisy, initial_size, maximum_size
    )

    threshold = otsu_threshold(img)

    segmented = (img >= threshold).astype(float)

    print("\nOtsu Threshold:", threshold)

    print(
        "Median Filter PSNR:",
        psnr(img, ordinary), "dB"
    )

    print(
        "Adaptive Median Filter PSNR:",
        psnr(img, adaptive), "dB"
    )

    save_image("original.png", img)
    save_image("noisy.png", noisy)
    save_image("median.png", ordinary)
    save_image("adaptive_median.png", adaptive)
    save_image("otsu.png", segmented)

    images = [
        img, noisy, ordinary, adaptive, segmented
    ]

    titles = [
        "Original",
        "Noisy",
        "Median Filter",
        "Adaptive Median",
        "Otsu Segmentation"
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
