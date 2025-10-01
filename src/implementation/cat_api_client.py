from io import BytesIO
from typing import List, Optional

import numpy as np
import requests
from PIL import Image

from src.implementation.cat_image import CatImage


class CatAPIClient:
    def __init__(self, api_key: Optional[str] = None, url: str = None):
        self.api_key = api_key
        self.url = url

    def fetch_cats(self, limit: int) -> List[CatImage]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        params = {"limit": limit}
        response = requests.get(self.url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        result = []
        for entry in data:
            image_url = entry.get("url")
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            img_pil = Image.open(BytesIO(img_response.content)).convert("RGB")
            img_np = np.array(img_pil)
            result.append(CatImage(image=img_np, url=image_url))
        return result
