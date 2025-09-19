from ImageClassification.entity import TrainingConfig
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from pathlib import Path

class Training:
    """
    Handles the training process for the image classification model.
    Includes methods for loading the base model, preparing data generators,
    training the model, and saving the trained model.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initializes the Training object with the given configuration.

        Args:
            config (TrainingConfig): Configuration dataclass for training.
        """
        self.config = config
    
    def get_base_model(self):
        """
        Loads the updated base model from the specified path and compiles it.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"]
        )
    
    def train_valid_generator(self):
        """
        Prepares the training and validation data generators.
        Applies data augmentation to the training generator if specified in the config.
        """
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20  # 20% of data for validation
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Exclude channels
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation data generator (no augmentation)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training data generator (with augmentation if enabled)
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the given Keras model to the specified path.

        Args:
            path (Path): Path to save the model.
            model (tf.keras.Model): The model to save.
        """
        model.save(path)

    def train(self, callback_list: list):
        """
        Trains the model using the prepared generators and callbacks.
        Saves the trained model after training.

        Args:
            callback_list (list): List of Keras callbacks to use during training.
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )