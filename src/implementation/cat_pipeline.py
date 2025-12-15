import asyncio
import os
from io import BytesIO
from pathlib import Path

import aiofiles
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from src.implementation.cat_api_client import CatAPIClient
from src.implementation.cat_image_processor import CatImageProcessor
from src.implementation.logger import PipelineLogger

pipeline_logger = PipelineLogger()
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv()


async def save_image_async(image_array: np.ndarray, file_path: Path):
    img_pil = Image.fromarray(image_array.astype(np.uint8))

    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(buffer.getvalue())


@pipeline_logger.timeit
async def run_pipeline_async(limit: int = 10):
    client = CatAPIClient(api_key=os.getenv("API_KEY"), url=os.getenv("BASE_URL"))
    processor = CatImageProcessor()

    pipeline_logger.logger.info("Fetching cat URLs...")
    pipeline_logger.logger.debug("debug!!!...")
    cat_data = await client.fetch_cats_urls(limit=limit)
    urls_with_indices = [(entry["url"], idx + 1) for idx, entry in enumerate(cat_data)]

    pipeline_logger.logger.info("Downloading images...")
    cat_images = await client.download_images(urls_with_indices)
    valid_images = [img for img in cat_images if not isinstance(img, Exception)]

    pipeline_logger.logger.info("Saving original images...")
    save_tasks = []
    for cat_image in valid_images:
        path = DATA_DIR / f"{cat_image.index}_original.png"
        save_tasks.append(save_image_async(cat_image.image, path))
    await asyncio.gather(*save_tasks)

    pipeline_logger.logger.info("Processing images...")
    processed_images = await processor.process_images_parallel(valid_images)

    pipeline_logger.logger.info("Saving processed images...")
    save_tasks = []
    for processed in processed_images:
        proc_path = DATA_DIR / f"{processed.index}_processed.png"
        save_tasks.append(save_image_async(processed.image.astype(np.uint8), proc_path))
    await asyncio.gather(*save_tasks)

    pipeline_logger.logger.info(
        f"Pipeline completed. Saved {len(valid_images)} images to {DATA_DIR}"
    )


if __name__ == "__main__":
    asyncio.run(run_pipeline_async(limit=5))
