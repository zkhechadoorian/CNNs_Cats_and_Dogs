import tensorflow as tf
from pathlib import Path
from ImageClassification.entity import EvaluationConfig
from ImageClassification.utils import save_json

class Evaluation:
    """
    Handles the evaluation of a trained Keras model on a validation dataset.
    Computes and saves evaluation metrics such as loss and accuracy.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the Evaluation object with the given configuration.

        Args:
            config (EvaluationConfig): Configuration dataclass for evaluation.
        """
        self.config = config

    def _valid_generator(self):
        """
        Prepares the validation data generator using ImageDataGenerator.
        Sets up the generator to read images from the validation split.
        """
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30  # 30% of data used for validation
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Exclude channels
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Create the validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Flow images from directory for validation
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Loads a Keras model from the specified file path.

        Args:
            path (Path): Path to the saved Keras model.

        Returns:
            tf.keras.Model: Loaded Keras model.
        """
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """
        Loads the model, prepares the validation generator, and evaluates the model.
        Stores the evaluation score (loss and accuracy) in self.score.
        """
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    def save_score(self):
        """
        Saves the evaluation scores (loss and accuracy) to a JSON file.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
