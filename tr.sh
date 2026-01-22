#!/usr/bin/env bash

# rm -rf work_dirs
# rm -rf /home/dell/gitrepos/MdaCD/work_dirs

python tools/copy_clip_files.py LEVIRCD penalty_10_mask
bash tools/general/train.sh configs/MdaCD_LEVIRCD.py 1 --work-dir /home/dell/gitrepos/MdaCD/work_dirs/MdaCD_LEVIRCD_4

python tools/copy_clip_files.py SYSUCD penalty_100_mask
bash tools/general/train.sh configs/MdaCD_SYSUCD.py 1 --work-dir /home/dell/gitrepos/MdaCD/work_dirs/MdaCD_SYSUCD_4

