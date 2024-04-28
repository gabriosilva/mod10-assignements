import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2 as cv

from pipeline import ProcessingPipeline
import processors as pr
import data_preparators as dp
import augmenters as ag

from PIL import Image


class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        imarray = np.random.rand(10,10,3) * 255
        self.test_image = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        self.test_image = np.array(imarray.astype('uint8'))

    def test_gaussian_blur_processor(self):
        processor = pr.GaussianBlurProcessor((5, 5))
        blurred_image = processor.process(self.test_image.copy())
        expected_result = [[[116, 109, 157], [123, 115, 168], [135, 130, 173], [141, 133, 149], [137, 109, 124], [127, 96, 124], [121, 115, 132], [113, 130, 126], [100, 122, 113], [93, 114, 108]], [[108, 95, 147], [117, 100, 158], [132, 118, 167], [139, 130, 148], [136, 114, 123], [123, 99, 116], [114, 108, 122], [111, 120, 125], [102, 116, 124], [96, 111, 124]], [[104, 80, 127], [112, 81, 139], [122, 100, 151], [126, 129, 140], [128, 132, 119], [118, 116, 110], [111, 111, 114], [118, 113, 129], [121, 115, 146], [118, 117, 154]], [[120, 88, 123], [118, 85, 131], [109, 98, 139], [103, 132, 132], [114, 147, 123], [121, 136, 123], [122, 125, 129], [137, 123, 137], [154, 130, 150], [160, 137, 158]], [[129, 111, 139], [123, 106, 141], [104, 112, 139], [92, 136, 134], [110, 143, 135], [131, 128, 139], [134, 115, 137], [142, 116, 129], [166, 130, 129], [180, 139, 132]], [[115, 129, 152], [115, 126, 151], [107, 130, 144], [99, 144, 139], [114, 140, 138], [134, 114, 130], [132, 94, 114], [126, 94, 103], [142, 106, 108], [156, 115, 114]], [[104, 125, 156], [107, 125, 154], [110, 132, 149], [110, 148, 142], [120, 150, 129], [135, 126, 107], [129, 102, 86], [111, 92, 86], [109, 92, 106], [114, 94, 120]], [[119, 105, 151], [118, 107, 150], [120, 117, 151], [127, 138, 150], [131, 152, 132], [138, 144, 102], [132, 122, 84], [110, 105, 88], [101, 98, 109], [103, 96, 121]], [[151, 103, 143], [147, 104, 143], [146, 108, 152], [148, 120, 161], [140, 139, 148], [132, 146, 118], [124, 131, 98], [112, 116, 98], [115, 112, 107], [124, 112, 113]], [[168, 110, 139], [164, 110, 140], [162, 108, 152], [159, 112, 166], [142, 130, 157], [124, 142, 127], [116, 132, 107], [111, 119, 103], [125, 118, 107], [138, 121, 110]]]
        self.assertFalse(np.array_equal(blurred_image, self.test_image))
        np.testing.assert_array_equal(blurred_image, expected_result)

    def test_invert_processor(self):
        processor = pr.InvertProcessor()
        inverted_image = processor.process(self.test_image.copy())
        self.assertFalse(np.array_equal(inverted_image, self.test_image))
        np.testing.assert_array_equal(inverted_image, 255 - self.test_image)

    def test_mean_adaptive_threshold_processor(self):
        processor = pr.MeanAdaptiveThresholdProcessor(3, 5)
        processed_image = processor.process(cv.cvtColor(self.test_image, cv.COLOR_BGR2GRAY))
        unique_values = np.unique(processed_image)
        self.assertFalse(np.array_equal(processed_image, self.test_image))
        self.assertTrue(np.array_equal(unique_values, [0, 255]))

    def test_resize_data_preparator(self):
        preparator = dp.ResizeDataPreparator((5, 5))
        resized_image = preparator.prepare_data(self.test_image.copy())
        self.assertFalse(np.array_equal(resized_image, self.test_image))
        self.assertEqual(resized_image.shape, (5, 5, 3))

    def test_rotate_augmenter(self):
        augmenter = ag.RotateAugmenter(90)
        rotated_image = augmenter.augment(self.test_image.copy())
        expected_results = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[240, 133, 105], [64, 118, 62], [23, 146, 236], [148, 224, 176], [234, 182, 254], [236, 179, 8], [115, 136, 228], [118, 70, 149], [149, 16, 123], [219, 248, 245]], [[30, 163, 36], [53, 41, 166], [144, 67, 133], [207, 101, 224], [205, 179, 25], [202, 57, 88], [6, 17, 173], [52, 108, 95], [170, 200, 71], [130, 57, 24]], [[203, 117, 199], [111, 252, 26], [105, 16, 176], [215, 178, 75], [93, 111, 227], [13, 50, 4], [177, 96, 45], [54, 132, 6], [25, 4, 237], [34, 209, 48]], [[198, 221, 249], [80, 92, 145], [75, 30, 81], [56, 242, 114], [166, 110, 228], [186, 43, 132], [14, 110, 79], [241, 186, 64], [245, 74, 61], [126, 163, 93]], [[22, 5, 212], [171, 53, 32], [9, 72, 30], [146, 150, 145], [150, 146, 166], [218, 2, 91], [125, 57, 64], [242, 175, 54], [21, 140, 149], [37, 250, 121]], [[144, 236, 18], [177, 15, 170], [249, 154, 188], [187, 245, 63], [34, 76, 145], [184, 220, 248], [47, 240, 188], [69, 203, 47], [195, 190, 230], [157, 3, 88]], [[97, 201, 134], [173, 91, 111], [24, 248, 119], [1, 172, 68], [73, 157, 109], [17, 177, 115], [79, 101, 53], [211, 160, 222], [124, 90, 239], [222, 74, 216]], [[111, 227, 245], [156, 157, 240], [209, 24, 213], [149, 5, 211], [4, 76, 168], [205, 145, 103], [136, 150, 186], [81, 97, 150], [144, 46, 36], [240, 203, 160]], [[138, 108, 164], [144, 4, 157], [35, 50, 94], [182, 73, 46], [164, 108, 154], [156, 31, 216], [60, 238, 156], [67, 5, 193], [33, 182, 100], [245, 59, 242]]]
        np.testing.assert_array_equal(rotated_image, expected_results)
        self.assertFalse(np.array_equal(rotated_image, self.test_image))
        self.assertEqual(rotated_image.shape, (10, 10, 3))

    def test_flip_horizontal(self):
        augmenter = ag.FlipAugmenter(1)  # 1 for horizontal flip
        flipped_image = augmenter.augment(self.test_image.copy())
        expected_flipped = np.fliplr(self.test_image.copy())
        np.testing.assert_array_equal(flipped_image, expected_flipped)

    def test_flip_vertical(self):
        augmenter = ag.FlipAugmenter(0)  # 0 for vertical flip
        flipped_image = augmenter.augment(self.test_image.copy())
        expected_flipped = np.flipud(self.test_image.copy())
        np.testing.assert_array_equal(flipped_image, expected_flipped)

    def test_flip_both_axes(self):
        augmenter = ag.FlipAugmenter(-1)  # -1 for flipping both axes
        flipped_image = augmenter.augment(self.test_image.copy())
        expected_flipped = np.flipud(self.test_image.copy())
        expected_flipped = np.fliplr(expected_flipped)
        np.testing.assert_array_equal(flipped_image, expected_flipped)

    def test_pipeline_integration(self):
        pipeline = ProcessingPipeline()
        pipeline.add_processor(pr.GaussianBlurProcessor((5, 5)))
        pipeline.add_processor(pr.InvertProcessor())
        pipeline.add_data_preparator(dp.ResizeDataPreparator((5, 5)))
        pipeline.add_augmenter(ag.RotateAugmenter(90))
        pipeline.add_augmenter(ag.RotateAugmenter(180))
        
        final_images = pipeline.run(self.test_image.copy())
        self.assertTrue(len(final_images) == 3)
        for img in final_images:
            self.assertEqual(img.shape, (5, 5, 3))

if __name__ == '__main__':
    unittest.main()
