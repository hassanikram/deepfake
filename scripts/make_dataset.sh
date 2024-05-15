#!/usr/bin/env bash

echo ""
echo "-------------------------------------------------"
echo "| Index CELEB-DF  dataset                       |"
echo "-------------------------------------------------"

python index_celebdf.py --source E:/MastersProject/Celeb-DF-v2



echo ""
echo "-------------------------------------------------"
echo "| Extract faces from CELEB-DF                   |"
echo "-------------------------------------------------"
python extract_faces.py --source E:/MastersProject/Celeb-DF-v2 --videodf E:/MastersProject/fakedetector-main/fakedetector-main/data/celebdf_videos.pkl --facesfolder E:/MastersProject/fakedetector-main/fakedetector-main/faces_output --facesdf E:/MastersProject/fakedetector-main/fakedetector-main/faces_df --checkpoint E:/MastersProject/fakedetector-main/fakedetector-main/tmp_per_video_output


