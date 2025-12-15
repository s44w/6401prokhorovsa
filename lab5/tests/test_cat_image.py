import unittest

import numpy as np

from implementation.cat_image import CatImage
from implementation.image_processing import ImageProcessing


class TestCatImage(unittest.TestCase):
    def test_image_addition(self):
        """Тест сложения изображений"""
        img1 = CatImage(image=np.array([[1, 2, 3], [3, 4, 5]], dtype=np.uint8), index=1)
        img2 = CatImage(image=np.array([[1, 2, 3], [3, 4, 5]], dtype=np.uint8), index=2)

        result = img1 + img2

        self.assertIsInstance(result, CatImage)
        np.testing.assert_array_equal(result.image, np.array([[2, 4, 6], [6, 8, 10]]))

    def test_convolution(self):
        """Тест свертки"""
        processor = ImageProcessing()
        test_image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        kernel = np.array([[1, 0, -1], [0, 0, -1], [1, 0, 0]], dtype=np.float32)

        result = processor._convolution2d(test_image, kernel)

        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.dtype, np.float32)
