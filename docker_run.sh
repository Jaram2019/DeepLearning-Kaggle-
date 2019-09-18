# Shell script for running Docker with TF2 image

sudo docker run \
    -d \
    -p 9090:8888 \
    -v /home/jaram/Desktop/19_2_Kagge_Study:/tf/kaggle_study \
    --name 19_2_kaggle_study_tf2 \
    --restart always \
    --runtime nvidia \
    tensorflow/tensorflow:2.0.0rc0-gpu-py3-jupyter
