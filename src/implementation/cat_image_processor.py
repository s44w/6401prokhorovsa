import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List

from image_processing import ImageProcessing

from src.implementation.cat_image import CatImage

logger = logging.getLogger("pipeline_logger")


class CatImageProcessor(ImageProcessing):
    def edge_detection_cat(self, cat_image: CatImage) -> CatImage:
        logger.info(f"Convolution for image {cat_image.index} started (PID {os.getpid()})")

        if cat_image.image is None:
            raise ValueError("CatImage has no image data")

        edges = self.edge_detection(cat_image.image)
        logger.info(f"Convolution for image {cat_image.index} finished (PID {os.getpid()})")
        return CatImage(image=edges, url=cat_image.url, index=cat_image.index)

    async def process_images_parallel(self, cat_images: List[CatImage]) -> List[CatImage]:
        logger.info("Starting parallel image processing...")

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(executor, self.edge_detection_cat, cat_image)
                for cat_image in cat_images
            ]
            results = await asyncio.gather(*tasks)

        logger.info("Parallel image processing completed")
        return results
