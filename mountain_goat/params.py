import os
PACKAGE_DIR=os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PACKAGE_DIR)

if __name__ == '__main__':
    print(BASE_DIR)
