import requests

import wget
from pathlib import Path
import pickle
from tqdm.auto import tqdm


def read_file(filepath):
    # Load the pickle file
    with open(filepath, "rb") as file:
        output = pickle.load(file)

    return output


def download_file(url, name):
    """Download a file from an specific url

    :param url: URL where the specific object is placed
    :param name: name of output file
    """
    print(url, name)
    try:
        # try with requests first as get estimated
        # time for completion
        download(url, name)
    except:
        # only resort to wget if have problems for some
        # reason - package hasn't been maintained since
        # 2015 so don't want dependence on this
        wget.download(url, out=name)


def download(url, filename, params=None):
    r = requests.get(url, stream=True, allow_redirects=True, params=params)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))
    block_size = 1024
    path = Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = "(Unknown total file size)" if file_size == 0 else ""
    with tqdm(total=file_size, unit="B", unit_scale=True, desc=desc) as progress_bar:
        with path.open("wb") as file:
            for data in r.iter_content(block_size):
                if data:  # filter out keep-alive new chunks
                    progress_bar.update(len(data))
                    file.write(data)
    return path
