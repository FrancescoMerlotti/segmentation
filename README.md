# Segmentation script

## Installation

1. Create a `conda` environment
    ``` bash
        conda create -n segmentation python=3.10
        conda activate segmentation
    ```

2. Install the required packages
    ``` bash
        pip install opencv-python matplotlib pandas scipy
        pip install torch torchvision
        pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

3. Download the weights of the model
    ``` bash
        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    ```

4. For logging
    ``` bash
        pip install git+https://github.com/FrancescoMerlotti/tronco.git
    ```

## Example

To run the example:
``` bash
    python segmentation.py --example
    python analyse.py --filename data/Segmented-Example.csv
```

- `segmentation.py` produces the segmentation information, stores it in a `.csv` file in the `data/` folder and produces a segmented image stored in the `processed/` folder.
- `analyse.py` loads the information stored in the selected `.csv` file in the `data/` folder, produces some statistically relevant quantities, and produces a histogram stored in the `plots/` folder.

Pre-processing of the image might be needed to increase the performance of the segmentation model. In particular increasing contrast could be helpful. Thus, run

``` bash
    python preprocess.py --filename preprocess/Test.jpg
```


## Statistically relevant information

1. Mean
2. Median
3. Mode
4. $95^{\rm th}$ percentile
5. Skewness
6. Interquartile range: explains where the middle 50% of the data points fall, effectively ignoring the extreme outliers appearing in the tail.