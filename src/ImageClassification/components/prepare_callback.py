import tensorflow as tf
tf.config.run_functions_eagerly(True)
import time
import os
from ImageClassification.entity import PrepareCallbacksConfig


class PrepareCallback:
    """
    Handles the creation of Keras callbacks for model training,
    including TensorBoard logging and model checkpointing.
    """

    def __init__(self, config: PrepareCallbacksConfig):
        """
        Initializes the PrepareCallback object with the given configuration.

        Args:
            config (PrepareCallbacksConfig): Configuration dataclass for callback preparation.
        """
        self.config = config

    
    @property
    def _create_tb_callbacks(self):
        """
        Creates a TensorBoard callback with a unique log directory based on the current timestamp.

        Returns:
            tf.keras.callbacks.TensorBoard: TensorBoard callback instance.
        """
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        """
        Creates a ModelCheckpoint callback to save the best model during training.

        Returns:
            tf.keras.callbacks.ModelCheckpoint: ModelCheckpoint callback instance.
        """
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )


    def get_tb_ckpt_callbacks(self):
        """
        Returns a list containing both TensorBoard and ModelCheckpoint callbacks.

        Returns:
            list: [TensorBoard callback, ModelCheckpoint callback]
        """
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]