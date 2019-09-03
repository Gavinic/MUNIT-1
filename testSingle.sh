#! /usr/bin/bash
CUDA_VISIBLE_DEVICES=2 python test_single.py \
       --config configs/UMID_REAL_256_noMalay.yaml \
       --input /home/hdd/zixin/MUNIT/datasets/testImage/599.jpg \
       --output_folder /home/hdd/zixin/MUNIT/outputs/Philipplines/image \
       --checkpoint /home/hdd/zixin/MUNIT/outputs/UMID_REAL_256_noMalay/checkpoints/gen_00270000.pt \
       --a2b 1 \
#       # a2b closed = b2a
#       --a2b 1  # a2b open
#/home/hdd/zixin/DataSets/Philip/test_fake
#/home/hdd/zixin/DataSets/Philip/test_real/00008.jpg

