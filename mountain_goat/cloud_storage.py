###############################################################################
# Download blob
###############################################################################
from google.cloud import storage

BUCKET_NAME = "my-bucket"

storage_filename = "data/raw/train_1k.csv"
local_filename = "train_1k.csv"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob(storage_filename)
blob.download_to_filename(local_filename)

###############################################################################
#Upload blob
###############################################################################
storage_filename = "models/random_forest_model.joblib"
local_filename = "model.joblib"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob(storage_filename)
blob.upload_from_filename(local_filename)
