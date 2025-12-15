import unittest
from unittest.mock import AsyncMock, patch

from src.implementation.cat_api_client import CatAPIClient


class TestCatAPIClient(unittest.IsolatedAsyncioTestCase):
    @patch("aiohttp.ClientSession.get")
    async def test_fetch_cats_urls(self, mock_get):
        """Минимальный тест апи"""
        mock_response = AsyncMock()
        mock_response.json.return_value = [{"url": "test_cat.jpg"}]
        mock_get.return_value.__aenter__.return_value = mock_response

        client = CatAPIClient(api_key="key123", url="https://api.cats.com")
        result = await client.fetch_cats_urls(limit=1)

        self.assertEqual(result[0]["url"], "test_cat.jpg")


if __name__ == "__main__":
    unittest.main()
