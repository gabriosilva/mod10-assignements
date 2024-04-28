import cv2 as cv
import numpy as np

class DataPreparator:
    """Base class for data preparation operations.

    This class provides a framework for preparing data and should be subclassed
    to implement specific data preparation techniques.
    """

    def __init__(self):
        """Initialize the data preparator."""
        pass

    def prepare_data(self, data):
        """Prepare the data. This method should be overridden in subclasses.

        Args:
            data (numpy.ndarray): The original data to be prepared.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Returns:
            numpy.ndarray: The prepared data.
        """
        raise NotImplementedError("This method should be implemented in child classes.")

class ResizeDataPreparator(DataPreparator):
    """A data preparator that resizes image data to a specified size.

    Attributes:
        size (tuple): The target size (height, width) for resizing.
    """

    def __init__(self, size):
        """Initialize the ResizeDataPreparator with a specific target size.

        Args:
            size (tuple): The target size (height, width) as a tuple.
        """
        super().__init__()
        self.size = size

    def prepare_data(self, data):
        """Resize the image data to the specified size.

        Args:
            data (numpy.ndarray): The image data to be resized.

        Returns:
            numpy.ndarray: The resized image.
        """
        return cv.resize(data, self.size, interpolation=cv.INTER_LINEAR)

class NormalizeOpencvImage(DataPreparator):
    """A data preparator that normalizes image data using OpenCV's normalization function."""

    def __init__(self):
        """Initialize the normalization preparator."""
        super().__init__()

    def prepare_data(self, data):
        """Normalize the image data to the range [0, 255].

        Args:
            data (numpy.ndarray): The image data to be normalized.

        Returns:
            numpy.ndarray: The normalized image.
        """
        return cv.normalize(data, None, 0, 255, cv.NORM_MINMAX)
