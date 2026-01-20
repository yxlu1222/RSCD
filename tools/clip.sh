python general/clip_inference.py --src_path /home/dell/gitrepos/MdaCD/Dataset/SYSUCD \
                                 --split train val test \
                                 --img_split penalty_100_mask1 penalty_100_mask2 \
                                 --model_name ViT-B/16 \
                                 --class_names_path general/rscls.txt \
                                 --device cuda:0 \
                                 --tag 56_vit16

python general/clip_inference.py --src_path /home/dell/gitrepos/MdaCD/Dataset/LEVIRCD \
                                 --split train val test \
                                 --img_split penalty_10_mask1 penalty_10_mask2 \
                                 --model_name ViT-B/16 \
                                 --class_names_path general/rscls.txt \
                                 --device cuda:0 \
                                 --tag 56_vit16
