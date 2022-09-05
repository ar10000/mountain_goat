import os
from google.oauth2.service_account import Credentials

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PACKAGE_DIR)
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")

# Credentials file
CREDENTIAL_FILE = f"{BASE_DIR}/credentials.json"
CREDENTIAL = Credentials.from_service_account_file(CREDENTIAL_FILE)


if __name__ == '__main__':
    print(BASE_DIR)
