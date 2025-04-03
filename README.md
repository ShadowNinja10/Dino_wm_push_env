## Dataset

- **Small dataset:**  
  - Contains 2 examples for debugging purposes.
  - Located at: `./pushenv`
- **Environment variable:**  
  - To set the dataset directory, run:
    ```bash
    export DATASET_DIR=./
    ```

## Implementation

- **Dataset:**  
  - The file `dataset/pushenv_dset.py` creates the pushenv dataset object.
- **Environment Code:**  
  - The directory `env/pushenv` contains:
    - `push_env.py`
    - `pushenv_wrapper.py`
- **Shape Configuration:**  
  - shape is handled by the shape file is present in the dataset. According to the shape respective env is created