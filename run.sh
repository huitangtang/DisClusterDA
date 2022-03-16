#!/bin/bash 

python main.py  --data_path_source /data/domain_adaptation/OfficeHome/  --src Art  --data_path_target_tr /data/domain_adaptation/OfficeHome/  --tar_tr Product  --data_path_target_te /data/domain_adaptation/OfficeHome/  --tar_te Product  --num_classes 65  --pretrained  --print_freq 20  --seed 1

python main.py  --data_path_source /data/domain_adaptation/OfficeHome/  --src Art  --data_path_target_tr /data/domain_adaptation/OfficeHome/  --tar_tr Product  --data_path_target_te /data/domain_adaptation/OfficeHome/  --tar_te Product  --num_classes 65  --pretrained  --print_freq 20  --seed 2

python main.py  --data_path_source /data/domain_adaptation/OfficeHome/  --src Art  --data_path_target_tr /data/domain_adaptation/OfficeHome/  --tar_tr Product  --data_path_target_te /data/domain_adaptation/OfficeHome/  --tar_te Product  --num_classes 65  --pretrained  --print_freq 20  --seed 3


python main.py  --data_path_source /data/domain_adaptation/OfficeHome/  --src Product  --data_path_target_tr /data/domain_adaptation/OfficeHome/  --tar_tr Art  --data_path_target_te /data/domain_adaptation/OfficeHome/  --tar_te Art  --num_classes 65  --pretrained  --print_freq 20  --seed 1

python main.py  --data_path_source /data/domain_adaptation/OfficeHome/  --src Product  --data_path_target_tr /data/domain_adaptation/OfficeHome/  --tar_tr Art  --data_path_target_te /data/domain_adaptation/OfficeHome/  --tar_te Art  --num_classes 65  --pretrained  --print_freq 20  --seed 2

python main.py  --data_path_source /data/domain_adaptation/OfficeHome/  --src Product  --data_path_target_tr /data/domain_adaptation/OfficeHome/  --tar_tr Art  --data_path_target_te /data/domain_adaptation/OfficeHome/  --tar_te Art  --num_classes 65  --pretrained  --print_freq 20  --seed 3


