# Script to download a dataset from a Dataverse (https://dataverse.org/)
import os
import sys
import hashlib
from pyDataverse.api import NativeApi, DataAccessApi


def download_file_in_dataset(base_url, api_token, doi, src, dest):
    api = NativeApi(base_url, api_token)
    data_api = DataAccessApi(base_url, api_token)
    dataset = api.get_dataset(doi)
    files_list = dataset.json()['data']['latestVersion']['files']
    for dv_file in files_list:
        remote_path = get_directory_path(dv_file)
        if remote_path == src:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            filepath = download_file(data_api, dv_file, dest)
            verify_checksum(dv_file, dest)
            return
    raise ValueError(f"Unable to find path {src} within {doi}.")


def get_directory_path(dv_file):
    directory_label = dv_file.get("directoryLabel")
    filename = dv_file["dataFile"]["filename"]
    if directory_label:
       return f"{directory_label}/{filename}"
    return filename


def download_file(data_api, dv_file, filepath):
    file_id = dv_file["dataFile"]["id"]
    print("Downloading file {}, id {}".format(filepath, file_id))
    response = data_api.get_datafile(file_id)
    with open(filepath, "wb") as f:
        f.write(response.content)
    return filepath


def verify_checksum(dv_file, filepath):
    checksum = dv_file["dataFile"]["checksum"]
    checksum_type = checksum["type"]
    checksum_value = checksum["value"]
    if checksum_type != "MD5":
        raise ValueError(f"Unsupported checksum type {checksum_type}")

    with open(filepath, 'rb') as infile:
        hash = hashlib.md5(infile.read()).hexdigest()
        if checksum_value == hash:
            print(f"Verified file checksum for {filepath}.")
        else:
            raise ValueError(f"Hash value mismatch for {filepath}: {checksum_value} vs {hash} ")


def show_usage():
   print()
   print(f"Usage: python {sys.argv[0]} <dataverse_base_url> <doi>\n")
   print("To specify an API token set the DATAVERSE_API_TOKEN environment variable.")
   print()


if __name__ == '__main__':
    if len(sys.argv) != 5:
         show_usage()
         sys.exit(1)
    else:
         base_url = sys.argv[1]
         doi = sys.argv[2]
         source = sys.argv[3]
         dest = sys.argv[4]
         api_token = os.environ.get('DATAVERSE_API_TOKEN')
         download_file_in_dataset(base_url, api_token, doi, source, dest)

