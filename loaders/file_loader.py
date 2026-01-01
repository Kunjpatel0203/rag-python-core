import requests
import os

DOWNLOAD_DIR = "data/downloads"

def download_file(url: str):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    filename = url.split("/")[-1]
    path = os.path.join(DOWNLOAD_DIR, filename)

    r = requests.get(url)
    with open(path, "wb") as f:
        f.write(r.content)

    return path
