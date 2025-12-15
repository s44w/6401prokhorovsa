import numpy as np
from numpy.lib._stride_tricks_impl import sliding_window_view
from scipy import ndimage


class ImageProcessing:
    def __init__(self, image_data_type=None):
        self._image_data_type = np.float32 if not image_data_type else image_data_type

    def _convolution2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2
        image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        windows = sliding_window_view(image_padded, kernel.shape)
        result = np.sum(windows * kernel, axis=(-1, -2))
        return result

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image: Входное изображение (может быть цветным или чёрно-белым)
            kernel: Ядро свёртки (матрица)

        Returns:
            Изображение после применения свёртки
        """
        image = image.astype(self._image_data_type)
        kernel = kernel.astype(self._image_data_type)
        if image.ndim == 3:
            result = np.zeros_like(image, dtype=self._image_data_type)
            for channel in range(image.shape[2]):
                result[:, :, channel] = self._convolution2d(
                    image=image[:, :, channel], kernel=kernel
                )
            return result
        else:
            return self._convolution2d(image=image, kernel=kernel)

    def _rgb_to_grayscale(self, image: np.ndarray, weights: np.array = None) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Args:
            image: Входное RGB-изображение

        Returns:
            Одноканальное изображение в оттенках серого
        """
        if not weights:
            weights = np.array([0.299, 0.587, 0.114])
        return np.dot(image[..., :3], weights).astype(self._image_data_type)

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Args:
            image: Входное изображение
            gamma: Коэффициент гамма-коррекции (>0)

        Returns:
            Изображение после гамма-коррекции
        """
        table = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(self._image_data_type)
        return table[image]

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении с использованием оператора Собеля.

        Args:
            image: Входное изображение (RGB)

        Returns:
            Одноканальное изображение с выделенными границами
        """
        gray = self._rgb_to_grayscale(image)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = self._convolution(image=gray, kernel=sobel_x)
        grad_y = self._convolution(image=gray, kernel=sobel_y)

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return (gradient_magnitude * 255 / gradient_magnitude.max()).astype(self._image_data_type)

    def edge_detection2(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении с использованием оператора Собеля.

        Args:
            image: Входное изображение (RGB)

        Returns:
            Одноканальное изображение с выделенными границами
        """
        gray = self._rgb_to_grayscale(image)
        import cv2

        edges = cv2.Canny(gray, 100, 200)
        return edges

    def corner_detection(
        self, image: np.ndarray, k: float = 0.04, threshold: float = 0.01
    ) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении с использованием алгоритма Харриса.

        Args:
            image: Входное изображение (RGB)
            k: Коэффициент для алгоритма Харриса
            threshold: Порог для обнаружения углов

        Returns:
            Изображение с выделенными углами (красные точки)
        """
        gray = self._rgb_to_grayscale(image).astype(float)

        Ix = ndimage.sobel(gray, axis=1)  # производные
        Iy = ndimage.sobel(gray, axis=0)

        Ixx = ndimage.gaussian_filter(Ix**2, sigma=1)  # второго порядка
        Ixy = ndimage.gaussian_filter(Ix * Iy, sigma=1)
        Iyy = ndimage.gaussian_filter(Iy**2, sigma=1)

        det = Ixx * Iyy - Ixy**2
        trace = Ixx + Iyy

        R = det - k * trace**2  # определитель харриса

        corners = np.zeros_like(R)
        local_max = ndimage.maximum_filter(R, size=3) == R
        corners[local_max & (R > threshold * R.max())] = 1  # ищем максимумы (локальные)

        result = image.copy()
        result[corners.astype(bool)] = [255, 0, 0]
        return result

    def corner_detection2(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса (cv2.cornerHarris) для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки).
        """
        import cv2

        gray = self._rgb_to_grayscale(image)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        result = image.copy()
        result[dst > 0.01 * dst.max()] = [255, 0, 0]
        return result

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Заглушка для обнаружения окружностей. Требует сложной реализации.

        Args:
            image: Входное изображение (RGB)

        Returns:
            Исходное изображение без изменений
        """
        # Реализация обнаружения окружностей требует преобразования Хафа
        # и является вычислительно сложной операцией
        return image.copy()
