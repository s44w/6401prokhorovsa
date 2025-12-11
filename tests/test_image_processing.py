import unittest

import numpy as np

from src.implementation.cat_image import CatImage
from src.implementation.cat_image_processor import CatImageProcessor


class TestCatImageProcessor(unittest.TestCase):
    def test_edge_detection(self):
        """Тест обнаружения границ"""
        processor = CatImageProcessor()

        test_image = np.zeros((8, 8, 3), dtype=np.uint8)
        test_image[2:4, 2:4, :] = 255

        cat_image = CatImage(image=test_image)

        result = processor.edge_detection_cat(cat_image)

        self.assertIsInstance(result, CatImage)
        self.assertEqual(len(result.image.shape), 2)


if __name__ == "__main__":
    unittest.main()
