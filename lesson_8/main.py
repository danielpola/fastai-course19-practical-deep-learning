#from fastai import datasets as fastai_datasets
import os
import pdb
import requests

def request_file(req_url, destination_path, replace=False):
    """ Download file if not exists """ 
    if not replace and os.path.exists(destination_path):
        print(f"    Already exists: {destination_path}")
        return

    # To emulate a browser and avoid being blocked.
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    }

    r = requests.get(req_url, headers=headers)
    open(destination_path, 'wb').write(r.content)

    print(f"    Downloaded: {destination_path}")

def download_mnist(destination_folder):
    """ Download MNIST data to destination folder if files don't exsits. """

    # Source: Yann Lecun ;) http://yann.lecun.com/exdb/mnist/index.html
    common_url = 'http://yann.lecun.com/exdb/mnist/'
    for dataset_name in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
        print(f"Downloading file (if not previously downloaded): {dataset_name}")
        file_url = os.path.join(common_url, dataset_name)
        dest_file_path = os.path.join(destination_folder, dataset_name)

        request_file(file_url, dest_file_path)

def main():
    # -----------------------------------------
    # Prepare Folders and download datasets
    # -----------------------------------------

    # download dataset if not exists
    inputs_folder = 'inputs/'
    outputs_folder = 'outputs/'
    for f in [inputs_folder, outputs_folder]:
        os.makedirs(f, exist_ok=True)

    download_mnist(inputs_folder)

    # -----------------------------------------
    # 
    # -----------------------------------------

    pdb.set_trace()

if __name__ == "__main__":
    main()