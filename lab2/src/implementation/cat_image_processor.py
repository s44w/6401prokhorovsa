from image_processing import ImageProcessing

from src.implementation.cat_image import CatImage


class CatImageProcessor(ImageProcessing):
    def edge_detection_cat(self, cat_image: CatImage) -> CatImage:
        if cat_image.image is None:
            raise ValueError("CatImage has no image data for edge detection")
        edges = self.edge_detection(cat_image.image)
        return CatImage(image=edges, url=cat_image.url)
