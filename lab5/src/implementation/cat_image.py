from dataclasses import dataclass

import numpy as np


@dataclass
class CatImage:
    image: np.ndarray = None
    url: str = None
    index: int = None

    def __add__(self, other):
        if not isinstance(other, CatImage):
            return ValueError(f"Passed {type(other)}, needs CatImage object")
        if len(self.image):
            result_image = self.image + other.image
            return CatImage(image=result_image, index=self.index)

    def __sub__(self, other):
        if not isinstance(other, CatImage):
            return ValueError(f"Passed {type(other)}, needs CatImage object")
        if len(self.image):
            result_image = self.image - other.image
            return CatImage(image=result_image, index=self.index)

    def __str__(self):
        return (
            f"CatImage(url={self.url}, image_shape={None if not len(self.image) else self.image.shape}, "
            f"index=f{self.index})"
        )
