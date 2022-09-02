###############################################################################
# Download blob
###############################################################################
from google.cloud import storage
from mountain_goat.params import CREDENTIAL
import os



BUCKET_NAME = 'mountain_goat_dataset'

storage_filename_screenshots = "mountain_goat_screenshots" #cloud path
local_filename_screenshots = "../raw_data/mountain_goat_screenshots" #local path
storage_filename_ucsd = "mountain_goat_UCSD" #cloud path
local_filename_ucsd = "../raw_data/mountain_goat_UCSD" #local path

def get_data(data_location, dataset):
    '''
    data_location values: 'cloud' or 'local'
    dataset values: 'screenshots' or 'ucsd'
    '''
    if data_location == 'cloud':
        client = storage.Client(credentials=CREDENTIAL)
        bucket = client.bucket(BUCKET_NAME)
        if dataset == 'screenshots':
            blobs = bucket.list_blobs(prefix=storage_filename_screenshots)
            for blob in blobs:
                #print(blob.name)
                # blob = bucket.blob(storage_filename_screenshots)
                print(blob.name)
                #print(os.getcwd())
                blob.download_to_filename(f'raw_data/{blob.name}')
        elif dataset == 'ucsd':
            blob = bucket.blob(storage_filename_ucsd)
            blob.download_to_filename(local_filename_ucsd)
    elif data_location == 'local':
        if dataset == 'screenshots':
            blob = bucket.blob(storage_filename_screenshots)
            blob.download_to_filename(local_filename_screenshots)
        elif dataset == 'ucsd':
            blob = bucket.blob(storage_filename_ucsd)
            blob.download_to_filename(local_filename_ucsd)

###############################################################################
#Upload blob
###############################################################################
# storage_filename = "models/random_forest_model.joblib"
# local_filename = "model.joblib"

# client = storage.Client()
# bucket = client.bucket(BUCKET_NAME)
# blob = bucket.blob(storage_filename)
# blob.upload_from_filename(local_filename)


if __name__ == '__main__':
    get_data('cloud', 'screenshots')
    # for i in range(0,116):
    #     os.mkdir(f"raw_data/mountain_goat_screenshots/video{i}")
