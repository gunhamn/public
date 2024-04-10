from enum import Flag, auto
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2  # noqa E402


class DataMode(Flag):
    """
    Flags to define data modes -- mono or color, binary or float, all classes or
    one missing. Standard setup would be MONO | BINARY.

      - MONO | BINARY: Standard one-channel MNIST dataset. All classes
        represented. Binarized. Use for learning standard generative models,
        check coverage, etc.
      - MONO | BINARY | MISSING: Standard one-channel MNIST dataset, but one
        class taken out. Use for testing "anomaly detection". Binarized.
      - MONO: Standard one-channel MNIST dataset, All classes there. Use for
        testing coverage etc. Data represented by their float values (not
        binarized). Can be easier to learn, but does not give as easy a
        probabilistic understanding.
      - MONO | MISSING: Standard one-channel MNIST dataset, but one class taken
        out. Use for testing anomaly detection use-case. Data represented by
        their float values (not binarized). Can be easier to learn, but does not
        give as easy a probabilistic understanding.
      - COLOR [| BINARY | MISSING]: These are *STACKED* versions of MNIST, i.e., three
        color channels with one digit in each channel.
    """

    MONO = 0
    COLOR = auto()
    BINARY = auto()
    MISSING = auto()


class Binarize:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        return (x > self.threshold).to(x.dtype)


class Scale:
    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def __call__(self, x):
        return self.min + x / 255 * self.max


class StackedMNISTData(Dataset):
    """
    The class will provide examples of data by sampling uniformly from MNIST
    data. We can do this one-channel (black-and-white images) or multi-channel
    (*STACKED* data), in which the last dimension will be the "color channel" of
    the image. In this case, 3 channels is the most natural, in which case each
    channel is one color (e.g. RGB).

    In the RGB-case we use channel 0 counting the ones for the red channel,
    channel 1 counting the tens for the green channel, and channel 2 counting
    the hundreds for the blue.
    """

    def __init__(
        self,
        root: str | Path,
        mode: DataMode = DataMode.MONO | DataMode.BINARY,
        train: bool = True,
    ):
        transforms = [v2.ToTensor(), v2.ToDtype(torch.float32), Scale()]

        # Make binary?
        if mode & DataMode.BINARY:
            transforms.append(Binarize())

        self.transform = v2.Compose(transforms)

        dataset = MNIST(
            root,
            train=train,
            download=True,
            transform=(transforms),
        )

        data = dataset.data
        targets = dataset.targets

        # Drop specific digit?  --- will only do this from training data
        if train and (mode & DataMode.MISSING):
            data = data[targets != 8]
            targets = targets[targets != 8]

        # Make colorful?
        if mode & DataMode.COLOR:
            indices = np.random.choice(a=len(data), size=(len(data), 3))
            data = torch.stack([data[indices[:, i]] for i in range(3)], dim=3)
            targets = (
                100 * targets[indices[:, 0]]
                + 10 * targets[indices[:, 1]]
                + targets[indices[:, 2]]
            )

        self.data = data
        self.targets = targets

    def __getitem__(self, index) -> Any:
        return self.transform(self.data[index]), self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
