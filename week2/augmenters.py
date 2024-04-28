import cv2 as cv

class DataAugmenter:
    """
    Base class for data augmentation operations.

    This class provides a framework for augmenting data and should be subclassed
    to implement specific augmentation techniques.
    """
    def __init__(self):
        """Initialize the data augmenter."""
        pass

    def augment(self, data):
        """
        Augment the data. This method should be overridden in subclasses.

        Args:
            data (numpy.ndarray): The original data to be augmented.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Returns:
            numpy.ndarray: The augmented data.
        """
        raise NotImplementedError("This method should be implemented in child classes.")


class RotateAugmenter(DataAugmenter):
    """
    An augmenter that rotates the image data by a specified angle.

    Attributes:
        angle (int): The angle in degrees to rotate the image, counter-clockwise.
    """
    def __init__(self, angle):
        """
        Initialize the RotateAugmenter with a specific angle.

        Args:
            angle (int): The angle in degrees to rotate the image, counter-clockwise.
        """
        super().__init__()
        self.angle = angle

    def augment(self, data):
        """
        Rotate the image by the specified angle around the center.

        Args:
            data (numpy.ndarray): The image data to be rotated.

        Returns:
            numpy.ndarray: The rotated image.
        """
        if len(data.shape) > 2:
            rows, cols, _ = data.shape
        else:
            rows, cols = data.shape
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), self.angle, 1)
        return cv.warpAffine(data, M, (cols, rows))


class FlipAugmenter(DataAugmenter):
    """
    An augmenter that flips the image data either horizontally, vertically, or both.

    Attributes:
        flip_code (int): Specifies how to flip the array; 0 means flipping around the x-axis
        and positive value (for example, 1) means flipping around y-axis. Negative value
        (for example, -1) means flipping around both axes.
    """
    def __init__(self, flip_code):
        """
        Initialize the FlipAugmenter with a specific flip code.

        Args:
            flip_code (int): The code indicating how the image should be flipped.
        """
        super().__init__()
        self.flip_code = flip_code

    def augment(self, data):
        """
        Flip the image according to the flip code.

        Args:
            data (numpy.ndarray): The image data to be flipped.

        Returns:
            numpy.ndarray: The flipped image.
        """
        return cv.flip(data, self.flip_code)
