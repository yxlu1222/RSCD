#!/usr/bin/env bash

# python tools/copy_clip_files.py LEVIRCD penalty_10_mask
bash tools/general/test.sh LEVIR configs/MdaCD_LEVIRCD.py 1 /home/dell/gitrepos/MdaCD/work_dirs/MdaCD_LEVIRCD_4

# python tools/copy_clip_files.py SYSUCD penalty_100_mask
bash tools/general/test.sh SYSU configs/MdaCD_SYSUCD.py 1 /home/dell/gitrepos/MdaCD/work_dirs/MdaCD_SYSUCD_4

# 举例：生成可视化结果（自动查找权重文件）
CHECKPOINT=$(find work_dirs/MdaCD_LEVIRCD_4 -name 'best_mIoU_iter_*.pth' -type f | head -n 1)
if [ -f "$CHECKPOINT" ]; then
    bash tools/general/dist_test.sh configs/MdaCD_LEVIRCD.py \
        "$CHECKPOINT" \
        1 \
        --show-dir work_dirs/MdaCD_LEVIRCD_4/vis_result
else
    echo "Error: No checkpoint found in work_dirs/MdaCD_LEVIRCD_4"
fi 