# Project Road Segmentation


## Setup for TAs

### 0. Create Conda environment (not mandatory)

This project was built using external libraries and python 3.5. To avoid packages conflicts and 
run properly our code we recommend to create a new Anaconda ([download](https://www.anaconda.com/download/)) environment 
and setup the environment using the following commands.

```
conda create -n env_road python=3.5
source activate env_road
```

### 1. Install requirements

All needed packages are listed in `requirements.txt` file. Use `pip` command to install packages
```
pip install -r requirements.txt
```


### 2. Use

Due to the size of the data (train and test) they were not committed on github. Place extracted data
from kaggle in folder named `data`. To use pre-trained model please download model
using dropbox link ([Final model download]()). The final folder should have the following structure:
```
project_final
    |-- run.py
    |-- ml_utils
    |-- model
        |-- submission_model.ckpt.data-00000-of-00001
        |-- submission_model.ckpt.ckpt.meta
        |-- submission_model.ckpt.index
    |-- data
        |-- training
            |-- ...
        |-- test_set_images
            |-- ...
```


To generate submission file with model use the following command. The results will be 
saved as `submission_final.csv`.
```
python run.py --model model/submission_model.ckpt
```

To run full training (! really long even with GPU !) use command 
```
python run.py
```




## Use GPU for dev

Connect to cluster using ssh command. Replace [username] with your github username and [ip] with cluster IP (35.195.6.119)
```
ssh -i ~/.ssh/id_rsa username@ip
```

On the cluster run
```
sudo su
cd ../ML_course/projects/project2/project_road_segmentation/
export PATH=/home/patryk_oleniuk/miniconda3/bin/:$PATH
source activate my-amazing-working-gpu
jupyter notebook --allow-root
```

On ou local machine connect to http://[EXTERNAL_IP_ADDRESS]:8888 (same IP as connection one, e.g. http://35.195.6.119:8888). Fill the token with the one one your terminal:
```
Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=[TOKEN_HERE]
```
