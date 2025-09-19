from pathlib import Path
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from ImageClassification.entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    Handles the preparation of the base model for transfer learning.
    Loads a pre-trained model, modifies it for the current task, and saves the updated model.
    """

    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes the PrepareBaseModel object with the given configuration.

        Args:
            config (PrepareBaseModelConfig): Configuration dataclass for base model preparation.
        """
        self.config = config

    def get_base_model(self):
        """
        Loads the pre-trained VGG16 model with specified parameters.
        Saves the base model to the configured path.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Modifies the base model for transfer learning:
        - Optionally freezes all or some layers.
        - Adds a flatten and dense (softmax) output layer.
        - Compiles the model.

        Args:
            model (tf.keras.Model): The base model.
            classes (int): Number of output classes.
            freeze_all (bool): Whether to freeze all layers.
            freeze_till (int or None): Number of layers from the end to keep trainable.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: The modified and compiled model.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        # Add new classification head
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Updates the base model by adding a new classification head and (optionally) unfreezing layers.
        Saves the updated model to the configured path.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=3,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the given Keras model to the specified path.

        Args:
            path (Path): Path to save the model.
            model (tf.keras.Model): The model to save.
        """
        model.save(path)