#! /usr/bin/bash
   CUDA_VISIBLE_DEVICES=0 python train.py \
        --config=configs/UMID_REAL_400_noMalay.yaml \
       --resume \


