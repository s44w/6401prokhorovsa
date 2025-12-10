import asyncio
from io import BytesIO
from typing import List, Optional

import aiohttp
import numpy as np
from PIL import Image

from src.implementation.cat_image import CatImage


class CatAPIClient:
    def __init__(self, api_key: Optional[str] = None, url: str = None):
        self.api_key = api_key
        self.url = url

    async def fetch_cats_urls(self, limit: int) -> List[dict]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}

        async with aiohttp.ClientSession() as session:
            async with session.get(self.url, headers=headers, params={"limit": limit}) as response:
                response.raise_for_status()
                return await response.json()

    @staticmethod
    async def download_image(url: str, index: int) -> CatImage:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                img_data = await response.read()
                img_pil = Image.open(BytesIO(img_data)).convert("RGB")
                return CatImage(image=np.array(img_pil), url=url, index=index)

    async def download_images(self, urls_with_indices: List[tuple]) -> List[CatImage]:
        tasks = [self.download_image(url, index) for url, index in urls_with_indices]
        return await asyncio.gather(*tasks, return_exceptions=True)
