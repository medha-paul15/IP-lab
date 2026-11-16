
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "assignment9_outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(7)

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

def gaussian_kernel(size=3, sigma=1):
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x, y = i-center, j-center
            kernel[i, j] = np.exp(
                -(x*x+y*y)/(2*sigma*sigma)
            )

    return kernel / kernel.sum()

def convolution(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    result = np.zeros_like(img)
    kernel = kernel[::-1, ::-1]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(
                padded[i:i+kh, j:j+kw] * kernel
            )

    return result

def gaussian_filter(img, size=3, sigma=1):
    return convolution(img, gaussian_kernel(size, sigma))

def wiener_filter(img, size=3):
    radius = size // 2
    padded = np.pad(img, radius, mode="reflect")

    means = np.zeros_like(img)
    variances = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+size, j:j+size]
            means[i, j] = np.mean(window)
            variances[i, j] = np.var(window)

    noise_variance = np.mean(variances)
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            variance = variances[i, j]
            gain = max(variance-noise_variance, 0) / max(variance, 1e-12)

            result[i, j] = (
                means[i, j] +
                gain * (img[i, j]-means[i, j])
            )

    return result

def bilateral_filter(img, size=3, sigma_space=1.5, sigma_range=0.1):
    radius = size // 2
    padded = np.pad(img, radius, mode="reflect")
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+size, j:j+size]
            center = img[i, j]

            weighted_sum = 0
            weight_total = 0

            for a in range(size):
                for b in range(size):
                    spatial = np.exp(
                        -((a-radius)**2+(b-radius)**2) /
                        (2*sigma_space**2)
                    )

                    intensity = np.exp(
                        -((window[a, b]-center)**2) /
                        (2*sigma_range**2)
                    )

                    weight = spatial * intensity

                    weighted_sum += weight * window[a, b]
                    weight_total += weight

            result[i, j] = weighted_sum / weight_total

    return result

def local_binary_pattern(img):
    h, w = img.shape
    padded = np.pad(img, 1, mode="reflect")
    result = np.zeros((h, w), dtype=np.uint8)

    offsets = [
        (-1, -1), (-1, 0), (-1, 1), (0, 1),
        (1, 1), (1, 0), (1, -1), (0, -1)
    ]

    for i in range(h):
        for j in range(w):
            center = padded[i+1, j+1]
            code = 0

            for bit, (di, dj) in enumerate(offsets):
                if padded[i+1+di, j+1+dj] >= center:
                    code |= (1 << (7-bit))

            result[i, j] = code

    return result

def main():
    img = read_image()

    gaussian_noise = RNG.normal(0, 0.06, img.shape)
    speckle_noise = RNG.normal(0, 0.12, img.shape)

    gaussian_noisy = np.clip(img + gaussian_noise, 0, 1)
    speckle_noisy = np.clip(
        img + img * speckle_noise, 0, 1
    )

    lbp = local_binary_pattern(img)
    Image.fromarray(lbp).save(OUT / "lbp.png")

    noise_images = {
        "Gaussian": gaussian_noisy,
        "Speckle": speckle_noisy
    }

    filters = {
        "Wiener": wiener_filter,
        "Median": median_filter,
        "Gaussian": gaussian_filter,
        "Bilateral": bilateral_filter
    }

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(lbp, cmap="gray")
    axes[0, 1].set_title("LBP")

    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    axes[0, 4].axis("off")

    for row, (noise_name, noisy) in enumerate(
        noise_images.items(), start=1
    ):
        save_image(noise_name.lower() + "_noisy.png", noisy)

        axes[row, 0].imshow(noisy, cmap="gray")
        axes[row, 0].set_title(noise_name + " Noise")

        for col, (filter_name, filter_function) in enumerate(
            filters.items(), start=1
        ):
            filtered = filter_function(noisy, 3)

            save_image(
                noise_name.lower() + "_" +
                filter_name.lower() + ".png",
                filtered
            )

            axes[row, col].imshow(filtered, cmap="gray")
            axes[row, col].set_title(filter_name)

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
