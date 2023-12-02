# system packages
from pathlib import Path
import shutil
import urllib.request
import os
import zipfile
import pandas as pd
import sys
from tqdm import tqdm

# Set the url
URL = "https://github.com/MScaramuzzi/Assignment-2/raw/main/data.zip"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(download_path: Path, url: str):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=download_path, reporthook=t.update_to)

def download_dataset(download_path: Path, url: str):
    print("Downloading dataset...")
    download_url(url=url, download_path=download_path)
    print("Download complete!")

def extract_zip(download_path: Path, extract_path: Path):
    print("Extracting dataset... (it may take a while...)")
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed!")

dataset_name = 'data'

def retrieve_data():
    
    dataset_folder = Path.cwd()

    if not dataset_folder.exists():
        dataset_folder.mkdir(parents=True)

    dataset_zip_path = dataset_folder.joinpath("data.zip")
    dataset_path = dataset_folder.joinpath("data")

    if not dataset_zip_path.exists():
        download_dataset(dataset_zip_path, URL)

    if not dataset_path.exists():
        extract_zip(dataset_zip_path, dataset_folder)

    os.remove(dataset_zip_path)

def read_data():
    pass

