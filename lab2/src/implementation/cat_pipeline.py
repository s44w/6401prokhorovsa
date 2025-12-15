import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from src.implementation.cat_api_client import CatAPIClient
from src.implementation.cat_image import CatImage
from src.implementation.cat_image_processor import CatImageProcessor
from src.implementation.logger import PipelineLogger

pipeline_logger = PipelineLogger()
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

processor = CatImageProcessor()
load_dotenv()


@pipeline_logger.timeit
def run_pipeline(limit: int = 10):
    client = CatAPIClient(api_key=os.getenv("API_KEY"), url=os.getenv("BASE_URL"))
    processor = CatImageProcessor()

    cat_images = client.fetch_cats(limit=limit)
    print(f"Got {len(cat_images)} images")

    for idx, cat_image in enumerate(cat_images, start=1):
        orig_filename = f"{idx}_original.png"
        proc_filename = f"{idx}_processed.png"
        enh_filename = f"{idx}_enh.png"
        orig_path = DATA_DIR / orig_filename
        proc_path = DATA_DIR / proc_filename
        enh_path = DATA_DIR / enh_filename

        Image.fromarray(cat_image.image).save(orig_path)
        processed = processor.edge_detection_cat(cat_image)
        img_rgb = np.stack((processed.image,) * 3, axis=-1)

        img = CatImage(image=img_rgb)
        enhanced = cat_image - img
        Image.fromarray(enhanced.image.astype(np.uint8)).save(enh_path)

        Image.fromarray(processed.image.astype(np.uint8)).save(proc_path)

        pipeline_logger.logger.info(f"Saved {orig_path} and {proc_path}")

    print(f"Saved all images to {DATA_DIR}")


if __name__ == "__main__":
    run_pipeline(limit=20)
