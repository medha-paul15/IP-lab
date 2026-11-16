
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.util import random_noise

BASE = Path(__file__).resolve().parent
OUT = BASE / "assignment8_outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

def read_image():
    return np.array(
        Image.open(BASE / "photos" / "cat1.png").convert("L"),
        dtype=float
    ) / 255.0

def save_image(name, img):
    img = np.clip(img, 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(OUT / name)

def generate_noise(shape):
    uniform = RNG.uniform(-0.15, 0.15, shape)

    gaussian = RNG.normal(0, 0.08, shape)

    erlang = RNG.gamma(shape=3, scale=0.04, size=shape)
    erlang = erlang - 3 * 0.04

    exponential = RNG.exponential(scale=0.12, size=shape)
    exponential = exponential - 0.12

    rayleigh = RNG.rayleigh(scale=0.10, size=shape)
    rayleigh = rayleigh - 0.10 * np.sqrt(np.pi / 2)

    return {
        "Uniform": uniform,
        "Gaussian": gaussian,
        "Erlang": erlang,
        "Exponential": exponential,
        "Rayleigh": rayleigh
    }

def main():
    img = read_image()
    noises = generate_noise(img.shape)

    fig, axes = plt.subplots(5, 3, figsize=(13, 17))

    for row, (name, noise) in enumerate(noises.items()):
        noisy = np.clip(img + noise, 0, 1)

        save_image(name.lower() + ".png", noisy)

        axes[row, 0].imshow(noisy, cmap="gray")
        axes[row, 0].set_title(name + " Noise")
        axes[row, 0].axis("off")

        axes[row, 1].hist(noise.ravel(), bins=60, density=True)
        axes[row, 1].set_title(name + " Histogram")

        if name == "Gaussian":
            builtin = random_noise(
                img,
                mode="gaussian",
                mean=0,
                var=0.08**2,
                clip=True
            )

            save_image("builtin_gaussian.png", builtin)
            axes[row, 2].imshow(builtin, cmap="gray")
            axes[row, 2].set_title("Built-in Gaussian Noise")
            axes[row, 2].axis("off")
        else:
            axes[row, 2].text(
                0.5, 0.5,
                "No direct imnoise equivalent",
                ha="center", va="center"
            )
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "noise_comparison.png")
    plt.show()

    print("All noise images and histograms saved in:", OUT)
    print("Uniform: U(-0.15, 0.15)")
    print("Gaussian: mean=0, standard deviation=0.08")
    print("Erlang: k=3, scale=0.04")
    print("Exponential: scale=0.12")
    print("Rayleigh: sigma=0.10")

if __name__ == "__main__":
    main()
