import os
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
from ImageClassification.entity import DataIngestionConfig
from ImageClassification import logger
from ImageClassification.utils import get_size
from tqdm import tqdm


class DataIngestion:
    """
    Handles downloading, extracting, and cleaning the image dataset for training.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion object with configuration.

        Args:
            config (DataIngestionConfig): Configuration dataclass for data ingestion.
        """
        self.config = config

    def download_file(self):
        """
        Downloads the dataset file from the specified URL if it does not already exist.
        Logs the download status and file information.
        """
        logger.info("Trying to download file >>>>>")
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def _get_updated_list_of_files(self, list_of_files):
        """
        Filters the list of files to include only .jpg images of cats and dogs.

        Args:
            list_of_files (list): List of file names from the zip archive.

        Returns:
            list: Filtered list containing only cat and dog image files.
        """
        return [f for f in list_of_files if f.endswith(".jpg") and ("Cat" in f or "Dog" in f)]

    def _preprocess(self, zf: ZipFile, f: str, working_dir: str):
        """
        Extracts a file from the zip archive if it does not already exist,
        and removes it if it is empty (size 0).

        Args:
            zf (ZipFile): Open zip file object.
            f (str): File name to extract.
            working_dir (str): Directory to extract the file to.
        """
        target_filepath = os.path.join(working_dir, f)
        if not os.path.exists(target_filepath):
            zf.extract(f, working_dir)
        
        # Remove corrupted or empty files
        if os.path.getsize(target_filepath) == 0:
            os.remove(target_filepath)

    def unzip_and_clean(self):
        """
        Unzips the dataset and removes unwanted or empty files.
        Only cat and dog images are extracted and cleaned.
        """
        logger.info(f"unzipping file and removing unwanted files")
        with ZipFile(file=self.config.local_data_file, mode="r") as zf:
            list_of_files = zf.namelist()
            updated_list_of_files = self._get_updated_list_of_files(list_of_files)
            # Use tqdm for progress visualization
            for f in tqdm(updated_list_of_files):
                self._preprocess(zf, f, self.config.unzip_dir)
