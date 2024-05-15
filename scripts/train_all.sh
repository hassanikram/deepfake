#!/usr/bin/env bash
DEVICE=0

echo ""
echo "-------------------------------------------------"
echo "| Train Xception on Celeb-DF                    |"
echo "-------------------------------------------------"

python train_binclass.py --net Xception --traindb celebdf --valdb celebdf --celeb_faces_df_path E:/MastersProject/fakedetector-main/fakedetector-main/faces_df/faces_df.pkl --celeb_faces_dir E:/MastersProject/fakedetector-main/fakedetector-main/faces_output --face scale --size 224 --batch 32 --lr 1e-5 --valint 500 --patience 10 --maxiter 30000 --seed 41 --attention --device 0



echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetB4 on Celeb-DF              |"
echo "-------------------------------------------------"

python train_binclass.py --net EfficientNetB4 --traindb celebdf --valdb celebdf --celeb_faces_df_path E:/MastersProject/fakedetector-main/fakedetector-main/faces_df/faces_df.pkl --celeb_faces_dir E:/MastersProject/fakedetector-main/fakedetector-main/faces_output --face scale --size 224 --batch 32 --lr 1e-5 --valint 500 --patience 10 --maxiter 30000 --seed 41 --attention --device 0


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4 on Celeb-DF       |"
echo "-------------------------------------------------"

python train_binclass.py --net EfficientNetAutoAttB4 --traindb celebdf --valdb celebdf --celeb_faces_df_path E:/MastersProject/fakedetector-main/fakedetector-main/faces_df/faces_df.pkl --celeb_faces_dir E:/MastersProject/fakedetector-main/fakedetector-main/faces_output --face scale --size 224 --batch 32 --lr 1e-5 --valint 500 --patience 10 --maxiter 30000 --seed 41 --attention --device 0