###############################################################################
# Download blob
###############################################################################
from google.cloud import storage
from mountain_goat.params import CREDENTIAL, CREDENTIAL_FILE, PACKAGE_DIR, RAW_DATA_DIR
import os

def get_data(data_location, dataset):
    '''
    This function retrieves the data from the specified source
    * data_location values: 'cloud' or 'local'
    * dataset values: 'screenshots' or 'ucsd'
    AND populates the RAW_DATA_DIR
    '''
    # BUCKET_NAME = 'mountain_goat_dataset'
    BUCKET_NAME = os.environ.get('BUCKET_NAME')

    cloud_storage_filename_screenshots = "mountain_goat_screenshots"
    local_storage_filename_screenshots = os.path.join(RAW_DATA_DIR, "mountain_goat_screenshots")
    cloud_storage_filename_ucsd = "mountain_goat_UCSD"
    local_storage_filename_ucsd = os.path.join(RAW_DATA_DIR, "mountain_goat_UCSD")

    if data_location == 'cloud':
        client = storage.Client(credentials=CREDENTIAL) # >>> Returns:<google.cloud.storage.client.Client object at 0x103f6f8b0>
        bucket = client.bucket(BUCKET_NAME) # >>> Returns:<Bucket: mountain_goat_dataset>
        if dataset == 'screenshots':
            blobs = bucket.list_blobs(prefix=cloud_storage_filename_screenshots) # >>> Returns:<google.api_core.page_iterator.HTTPIterator object at 0x101004cd0>
            for index, blob in enumerate(blobs):
                # Create direc
                _, file_dir, file_name = blob.name.split('/')
                # print(blob.name)
                local_dir = os.path.join(local_storage_filename_screenshots, file_dir)
                # if not(os.path.exists(local_dir)):
                #     os.mkdir(local_dir)
                # local_path = os.path.join(local_dir, file_name)
                # blob.download_to_filename(local_path)
                # print(index, blob.name, end='\r')

                if not(os.path.exists(local_dir)):
                    os.mkdir(local_dir)
                local_path = os.path.join(local_dir, file_name)
                if not(os.path.exists(local_path)):
                    blob.download_to_filename(local_path)
                    print(index, blob.name, end='\r')
                print(index, blob.name, 'already exists',end='\r')

        elif dataset == 'ucsd':
            blob = bucket.blob(cloud_storage_filename_ucsd)
            # blob.download_to_filename(local_storage_filename_ucsd)
    elif data_location == 'local':
        if dataset == 'screenshots':
            blob = bucket.blob(cloud_storage_filename_screenshots)
            blob.download_to_filename(local_storage_filename_screenshots)
        elif dataset == 'ucsd':
            blob = bucket.blob(cloud_storage_filename_ucsd)
            blob.download_to_filename(local_storage_filename_ucsd)

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
