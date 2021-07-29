#train cityscape
python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset cityscapes --cv 0 --syncbn --apex --fp16 --crop_size 1024,2048 --bs_trn 1 --poly_exp 2 --lr 5e-3 --rmi_loss --max_epoch 175 --n_scales 0.5,1.0,2.0 --supervised_mscale_loss_wt 0.05 --snapshot ASSETS_PATH/seg_weights/ocrnet.HRNet_industrious-chicken.pth --arch ocrnet.HRNet_Mscale --result_dir logs/train_cityscapes/ocrnet.HRNet_Mscale_honored-agama_2021.07.27_07.48

#train boe
CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node=3 finetune_train.py --dataset boe-semantic-segmentation --cv 0 --apex --fp16 --crop_size 1024,2048 --bs_trn 1 --poly_exp 2 --lr 1e-3 --rmi_loss --max_epoch 20 --n_scales 0.5,1.0,2.0 --supervised_mscale_loss_wt 0.05 --snapshot ASSETS_PATH/seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth --arch ocrnet.HRNet_Mscale --result_dir logs/train_boe/ocrnet.HRNet_Mscale_honored-boe
python -m torch.distributed.launch --nproc_per_node=3 finetune_train.py --dataset boe-semantic-segmentation --cv 0 --apex --fp16  --pre_size 2048 --crop_size 1024,2048 --bs_trn 1 --poly_exp 2 --lr 1e-3 --rmi_loss --max_epoch 20 --n_scales 0.5,1.0,2.0 --supervised_mscale_loss_wt 0.05 --snapshot ASSETS_PATH/seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth --arch ocrnet.HRNet_Mscale --result_dir logs/train_boe/ocrnet.HRNet_Mscale_honored-boe

