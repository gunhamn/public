from itertools import zip_longest
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import Tensor


def _calc_grid_size(
    num_images: int, num_across: int = None, num_down: int = None
) -> tuple[int, int]:
    if num_across is None and num_down is None:
        width = int(np.sqrt(num_images))
        height = int(np.ceil(float(num_images) / width))
    elif num_across is not None:
        width = num_across
        height = int(np.ceil(float(num_images) / width))
    else:
        height = num_down
        width = int(np.ceil(float(num_images) / height))
    return width, height


def tile_images(
    images: np.ndarray,
    num_across: int = None,
    num_down: int = None,
    file_name: str = None,
) -> np.ndarray:
    """
    Take as set of images in, and tile them. Input images are represented as
    numpy array with 3 or 4 dims: shape[0]: Number of images shape[1] +
    shape[2]: size of image shape[3]: If > 1, then this is the color channel

    Images: The np array with images num_across/num_down: Force layout of subfigs.
    If both arte none, we get a "semi-square" image show: do plt.show()
    filename: If not None we save to this filename. Assumes it is fully extended
    (including .png or whatever)
    """

    width, height = _calc_grid_size(
        num_images=images.shape[0], num_across=num_across, num_down=num_down
    )

    if len(images.shape) < 4:
        images = np.expand_dims(images, axis=-1)
    color_channels = images.shape[3]

    # Rescale
    images = images - np.min(np.min(np.min(images, axis=(1, 2))))
    images = images / np.max(np.max(np.max(images, axis=(1, 2))))

    # Build up tiled representation
    image_shape = images.shape[1:3]
    tiled_image = np.zeros(
        (height * image_shape[0], width * image_shape[1], color_channels),
        dtype=images.dtype,
    )
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        tiled_image[
            i * image_shape[0] : (i + 1) * image_shape[0],  # noqa E203
            j * image_shape[1] : (j + 1) * image_shape[1],  # noqa E203
            :,
        ] = img

    fig, ax = plt.subplots()
    if color_channels == 1:
        ax.imshow(tiled_image[:, :, 0], cmap="binary")
    else:
        ax.imshow(tiled_image.astype(np.float32))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    return tiled_image, fig, ax


def tile_pil_images(
    images: Sequence[Image.Image],
    num_across: int = None,
    num_down: int = None,
    file_name: str = None,
) -> Image.Image:
    """
    Take a set of images in, and tile them. Input images are represented as
    PIL images.
    """

    num_across, num_down = _calc_grid_size(
        num_images=len(images), num_across=num_across, num_down=num_down
    )

    img_width, img_height = images[0].size
    mode = images[0].mode

    tiled_image = Image.new(mode, (img_width * num_across, img_height * num_down))

    indices = np.indices((num_across, num_down))

    for index, img in enumerate(images):
        x = indices[0].flat[index] * img_width
        y = indices[1].flat[index] * img_height
        tiled_image.paste(img, (x, y))

    plt.Figure()
    plt.imshow(tiled_image, cmap="gray" if mode == "L" else None)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    return tiled_image


def tile_tv_images(
    images: Tensor,
    labels: Iterable[str] = (),
    num_across: int = None,
    num_down: int = None,
    file_name: str = None,
):
    """
    Tile and plot a tensor of images. 2D-tensors are plotted as grayscale, 4D as
    color.
    """

    num_across, num_down = _calc_grid_size(
        num_images=len(images), num_across=num_across, num_down=num_down
    )

    fig, axes = plt.subplots(nrows=num_down, ncols=num_across)
    mono = len(images.shape) == 3
    for img, ax, label in zip_longest(images, axes.flat, labels):
        ax.imshow(img, cmap="gray" if mono else None)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title=label)

    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    return fig, axes
