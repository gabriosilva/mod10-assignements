import cv2 as cv

class DataProcessor:
    """
    Base class for data processing operations.

    This class provides a framework for processing data and should be subclassed
    to implement specific data processing techniques.
    """

    def __init__(self):
        """Initialize the data processor."""
        pass

    def process(self, data):
        """
        Process the data. This method should be overridden in subclasses.

        Args:
            data (numpy.ndarray): The original data to be processed.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("This method should be implemented in child classes.")

class GaussianBlurProcessor(DataProcessor):
    """
    A data processor that applies Gaussian blur to the image data.

    Attributes:
        kernel_size (tuple): The kernel size for the Gaussian blur.
    """

    def __init__(self, kernel_size):
        """
        Initialize the GaussianBlurProcessor with a specific kernel size.

        Args:
            kernel_size (tuple): The kernel size for the Gaussian blur, specified as (height, width).
        """
        super().__init__()
        self.kernel_size = kernel_size

    def process(self, data):
        """
        Apply Gaussian blur to the image data.

        Args:
            data (numpy.ndarray): The image data to be blurred.

        Returns:
            numpy.ndarray: The blurred image.
        """
        return cv.GaussianBlur(data, self.kernel_size, 0)

class MeanAdaptiveThresholdProcessor(DataProcessor):
    """
    A data processor that applies mean adaptive thresholding to the image data.

    Attributes:
        block_size (int): Size of a pixel neighborhood that is used to calculate a threshold value.
        c (int): Constant subtracted from the mean or weighted mean.
    """

    def __init__(self, block_size, c):
        """
        Initialize the MeanAdaptiveThresholdProcessor with specific parameters.

        Args:
            block_size (int): Size of the block used to calculate the threshold.
            c (int): Constant subtracted from the calculated mean or weighted mean.
        """
        super().__init__()
        self.block_size = block_size
        self.c = c

    def process(self, data):
        """
        Apply mean adaptive thresholding to the image data.

        Args:
            data (numpy.ndarray): The image data to be thresholded.

        Returns:
            numpy.ndarray: The thresholded image.
        """
        _data = cv.cvtColor(data, cv.COLOR_BGR2GRAY) if len(data.shape) > 2 else data
        return cv.adaptiveThreshold(_data, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY, self.block_size, self.c)

class InvertProcessor(DataProcessor):
    """
    A data processor that inverts the image data.
    """

    def __init__(self):
        """Initialize the InvertProcessor."""
        super().__init__()

    def process(self, data):
        """
        Invert the image data.

        Args:
            data (numpy.ndarray): The image data to be inverted.

        Returns:
            numpy.ndarray: The inverted image.
        """
        return cv.bitwise_not(data)

class CannyProcessor(DataProcessor):
    """
    A data processor that applies Canny edge detection to the image data.

    Attributes:
        threshold1 (int): First threshold for the hysteresis procedure.
        threshold2 (int): Second threshold for the hysteresis procedure.
    """

    def __init__(self, threshold1, threshold2):
        """
        Initialize the CannyProcessor with specific thresholds.

        Args:
            threshold1 (int): Lower threshold.
            threshold2 (int): Upper threshold.
        """
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def process(self, data):
        """
        Apply Canny edge detection to the image data.

        Args:
            data (numpy.ndarray): The image data for edge detection.

        Returns:
            numpy.ndarray: The image with detected edges.
        """
        return cv.Canny(data, self.threshold1, self.threshold2)
