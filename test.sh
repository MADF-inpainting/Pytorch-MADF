#ln -s -f ../testset/place2_testset/images_1k/
#ln -s -f ../testset/place2_testset//
ln -s -f ../../dataset/val_large_1k
ln -s -f ../../dataset/mask_reverse/testing_mask_reverse_dataset/
ln -s -f ../../testset/place2_testset/test_large_1.2w
rm result_place2/*
rm gt_result_place2/*
#export LD_LIBRARY_PATH=~/zhumanyu/cuda-9.0/lib64/:~/zhumanyu/cudnn-7.1/cuda/lib64/:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=1 ~/zhumanyu/env/bin/python test.py
