import os
import cv2 as cv
import pandas as pd
import logging

from pipeline import ProcessingPipeline
import processors as pr
import data_preparators as dp
import augmenters as ag

INPUT_FOLDER = "pictures"
OUTPUT_FOLDER = "out"

logging.basicConfig(level=logging.INFO)


def load_pictures(input_folder):
    """
    Load images from the specified folder with extensions png, jpg, or jpeg.

    Args:
        input_folder (str): The directory from which to load images.

    Returns:
        tuple: A tuple containing:
            - List of loaded images as NumPy arrays.
            - List of file paths corresponding to the loaded images.
    """
    picture_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    logging.info(f"Found {len(picture_paths)} pictures in the folder {input_folder}.")
    return [cv.imread(path) for path in picture_paths], picture_paths


def initialize_pipeline():
    """
    Initialize the processing pipeline with specific processors, preparators, and augmenters.

    Returns:
        ProcessingPipeline: An instance of the ProcessingPipeline configured with necessary components.
    """
    pipeline = ProcessingPipeline()
    pipeline.add_processor(pr.GaussianBlurProcessor((5, 5)))
    pipeline.add_processor(pr.InvertProcessor())
    pipeline.add_processor(pr.CannyProcessor(100, 200))
    pipeline.add_data_preparator(dp.ResizeDataPreparator((400, 400)))
    pipeline.add_augmenter(ag.RotateAugmenter(90))
    pipeline.add_augmenter(ag.RotateAugmenter(45))
    pipeline.add_augmenter(ag.FlipAugmenter(1))
    pipeline.add_augmenter(ag.FlipAugmenter(0))
    return pipeline


def process_images(pipeline, pictures, picture_paths, output_folder):
    """
    Process each image using the provided pipeline and save the processed images.

    Args:
        pipeline (ProcessingPipeline): The image processing pipeline.
        pictures (list): List of images (as NumPy arrays) to process.
        picture_paths (list): List of file paths corresponding to each image.
        output_folder (str): Directory to save processed images.

    Returns:
        list: A list containing information about the processed images.
    """
    categories = {"articfox", "cat", "dog", "redpanda", "squirrel"}
    processed_info = []

    for idx, (image, path) in enumerate(zip(pictures, picture_paths)):
        category = path.split(os.sep)[-1].split("_")[0]
        if category not in categories:
            logging.warning(f"Category {category} not in the list. Skipping...")
            continue

        processed_images = pipeline.run(image)
        save_processed_images(
            processed_images, category, idx, output_folder, processed_info
        )

    return processed_info


def save_processed_images(
    processed_images, category, index, output_folder, processed_info
):
    """
    Save the processed images to the specified output folder and record their information.

    Args:
        processed_images (list): List of processed images.
        category (str): Category of the image.
        index (int): Index of the original image in the batch.
        output_folder (str): Folder to save the processed images.
        processed_info (list): List to append information about saved images.
    """
    types = ["original", "processed", "augmented"]
    for idx, image in enumerate(processed_images):
        file_name = f"{category}_{index}_{idx}.jpg"
        path = os.path.join(output_folder, file_name)
        cv.imwrite(path, image)
        processed_info.append((path, types[1 if idx == 0 else 2], category))


def save_to_csv(data, filename):
    """
    Save the data about processed images to a CSV file.

    Args:
        data (list): Data to be saved.
        filename (str): Name of the file to save the data.
    """
    df = pd.DataFrame(data, columns=["image", "type", "category"])
    df.to_csv(filename)
    logging.info(f"Saved processed data to {filename}")


def main():
    """
    Main function to load pictures, process them through the pipeline, and save the results.
    """
    pictures, picture_paths = load_pictures(INPUT_FOLDER)
    pipeline = initialize_pipeline()
    processed_info = process_images(pipeline, pictures, picture_paths, OUTPUT_FOLDER)
    save_to_csv(processed_info, "image_dataframe.csv")


if __name__ == "__main__":
    main()
