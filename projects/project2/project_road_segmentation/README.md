# Project Road Segmentation

For this choice of project task, we provide a set of satellite images acquired from GoogleMaps.
We also provide ground-truth images where each pixel is labeled as road or background. 

Your task is to train a classifier to segment roads in these images, i.e. assigns a label `road=1, background=0` to each pixel.

Submission system environment setup:

1. The dataset is available from the Kaggle page, as linked in the PDF project description

2. Obtain the python notebook `segment_aerial_images.ipynb` from this github folder,
to see example code on how to extract the images as well as corresponding labels of each pixel.

The notebook shows how to use `scikit learn` to generate features from each pixel, and finally train a linear classifier to predict whether each pixel is road or background. Or you can use your own code as well. Our example code here also provides helper functions to visualize the images, labels and predictions. In particular, the two functions `mask_to_submission.py` and `submission_to_mask.py` help you to convert from the submission format to a visualization, and vice versa.

3. As a more advanced approach, try `tf_aerial_images.py`, which demonstrates the use of a basic convolutional neural network in TensorFlow for the same prediction task.

Evaluation Metric:
 [https://www.kaggle.com/wiki/MeanFScore]


## Jupyter on cluster

Connect to cluster using ssh command. Replace [username] with your github username and [ip] with cluster IP (35.195.6.119)
```
ssh -i ~/.ssh/id_rsa username@ip
``

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
