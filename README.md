# deepfake

In the Scripts directory, we have .sh files for making datasets, model training, and testing 
I ran all the commands one by one on Anaconda Powershell 
first I created an environment using 

$ conda env create -f environment.yml

$ conda activate deepfake

than ran commands in make_data.sh and after that I edited the code in train_binclass.py and split.py according to  my dataset
3 models are trained 
Xception
EfficientNetB4 and
EfficientNetAutoAttB4

all the weights are stored in the /weights/binclass directory
then all the models were tested separately and the results are stored in /results directory

the model is overfitting so we have to fix that as well
the code is using GPU but due to library version issues I couldn't use my gpu I have Cuda installed and setup but still it's not working

we now need to fix the code following tasks are required from you: 
1. Try to fix the overfitting issue
2. Make an ensemble of all the models and make it run on GPU
3. Develop a code which has a UI that takes a video path as input and prints if the video is fake or real.
4. Integrate explainable AI (XAI) so that we can show our users why our model detected the video as fake.

The timeline is around 1 month we have to complete it before 10th June 2024

the whole working code is uploaded in this repo and you can find the dataset at the following link:
https://drive.google.com/file/d/1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj/view
