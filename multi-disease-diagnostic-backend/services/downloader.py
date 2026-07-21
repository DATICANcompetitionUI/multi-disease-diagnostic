import os
import gdown


def download_file(file_id: str, output: str):

    if os.path.exists(output):
        return

    print(f"Downloading {output}...")

    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        output,
        quiet=False,
    )