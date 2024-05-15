#!/usr/bin/env bash
DEVICE=0

echo ""
echo "-------------------------------------------------"
echo "| Test Xception on Celeb                        |"
echo "-------------------------------------------------"

python test_model.py --model_path weights/binclass/net-Xception_traindb-celebdf_face-scale_size-224_seed-41/bestval.pth  --testsets celebdf --celeb_faces_df_path E:/MastersProject/fakedetector-main/fakedetector-main/faces_df/faces_df.pkl --celeb_faces_dir E:/MastersProject/fakedetector-main/fakedetector-main/faces_output --device 0 --num_video 1000



echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetB4 on Celeb                 |"
echo "-------------------------------------------------"

python test_model.py --model_path weights/binclass/net-EfficientNetB4_traindb-celebdf_face-scale_size-224_seed-41/bestval.pth  --testsets celebdf --celeb_faces_df_path E:/MastersProject/fakedetector-main/fakedetector-main/faces_df/faces_df.pkl --celeb_faces_dir E:/MastersProject/fakedetector-main/fakedetector-main/faces_output --device 0 --num_video 1000



echo ""
echo "-------------------------------------------------"
echo "| Test EfficientNetAutoAttB4 on Celeb          |"
echo "-------------------------------------------------"

python test_model.py --model_path weights/binclass/net-EfficientNetAutoAttB4_traindb-celebdf_face-scale_size-224_seed-41/bestval.pth  --testsets celebdf --celeb_faces_df_path E:/MastersProject/fakedetector-main/fakedetector-main/faces_df/faces_df.pkl --celeb_faces_dir E:/MastersProject/fakedetector-main/fakedetector-main/faces_output --device 0 --num_video 1000

