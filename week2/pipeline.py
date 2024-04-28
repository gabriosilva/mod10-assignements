class ProcessingPipeline:
    """
    A flexible pipeline for processing data through a series of processors,
    data preparators, and augmenters.

    Attributes:
        processors (list): List of processor objects to process data.
        augmenters (list): List of augmenter objects to augment data.
        data_preparators (list): List of data preparator objects to prepare data before processing.
        processors_timeline (list): Timeline of data after each processing step.
        data_preparators_timeline (list): Timeline of data after each preparation step.
        augmenters_timeline (list): Timeline of data after each augmentation step.
    """

    def __init__(self):
        """
        Initializes the processing pipeline with empty lists for processors, augmenters,
        and data preparators, along with their timelines.
        """
        self.processors = []
        self.augmenters = []
        self.data_preparators = []
        self.processors_timeline = []
        self.augmenters_timeline = []
        self.data_preparators_timeline = []

    def add_processor(self, processor):
        """
        Adds a processor to the pipeline.

        Args:
            processor: A processor object to be added to the pipeline.
        """
        self.processors.append(processor)

    def add_data_preparator(self, data_preparator):
        """
        Adds a data preparator to the pipeline.

        Args:
            data_preparator: A data preparator object to be added to the pipeline.
        """
        self.data_preparators.append(data_preparator)

    def add_augmenter(self, augmenter):
        """
        Adds an augmenter to the pipeline.

        Args:
            augmenter: An augmenter object to be added to the pipeline.
        """
        self.augmenters.append(augmenter)

    def process(self, data):
        """
        Processes the given data through all processors in the pipeline.

        Args:
            data: The data to be processed.

        Returns:
            The processed data.
        """
        _data = data
        self.processors_timeline = [data]
        for processor in self.processors:
            _data = processor.process(_data)
            self.processors_timeline.append(_data)
        return _data

    def prepare_data(self, data):
        """
        Prepares the given data through all data preparators in the pipeline.

        Args:
            data: The data to be prepared.

        Returns:
            The prepared data.
        """
        _data = data
        self.data_preparators_timeline = [data]
        for data_preparator in self.data_preparators:
            _data = data_preparator.prepare_data(_data)
            self.data_preparators_timeline.append(_data)
        return _data

    def augment(self, data):
        """
        Augments the given data using all augmenters in the pipeline.

        Args:
            data: The data to be augmented.

        Returns:
            A list of augmented data variations.
        """
        _data_augmented = [data]
        for augmenter in self.augmenters:
            _data_augmented.append(augmenter.augment(data))
        return _data_augmented

    def run(self, data):
        """
        Runs the complete pipeline on the given data by preparing, processing,
        and augmenting it in sequence.

        Args:
            data: The data to be run through the pipeline.

        Returns:
            A list containing the augmented data arrays produced by the pipeline.
        """
        _data = self.prepare_data(data)
        _data = self.process(_data)
        _augmented_data_array = self.augment(_data)
        return _augmented_data_array

    # Clearing methods for processors, augmenters, data preparators, and their timelines.
    # These methods reset respective components to an empty state, useful for reconfiguring the pipeline dynamically.

    def clear_processors(self):
        """Clears all processors from the pipeline."""
        self.processors = []

    def clear_data_preparators(self):
        """Clears all data preparators from the pipeline."""
        self.data_preparators = []

    def clear_augmenters(self):
        """Clears all augmenters from the pipeline."""
        self.augmenters = []

    def clear_timeline(self):
        """Clears all timelines for processors, data preparators, and augmenters."""
        self.clear_processors_timeline()
        self.clear_data_preparators_timeline()
        self.clear_augmenters_timeline()
