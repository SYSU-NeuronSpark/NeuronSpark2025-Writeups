K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res

INPUT_SIZE=256
DATASET=pixabay
NAME=slbr_v1

CUDA_VISIBLE_DEVICES=7 python3  test.py \
  --nets slbr \
  --models slbr \
  --input-size ${INPUT_SIZE} \
  --crop_size ${INPUT_SIZE} \
  --test-batch 1 \
  --evaluate\
  --dataset_dir /home/mozihao/Dewatermarking/Data/pixabay/pixabay_256 \
  --preprocess resize \
  --no_flip \
  --name ${NAME} \
  --mask_mode ${MASK_MODE} \
  --k_center ${K_CENTER} \
  --dataset ${DATASET} \
  --resume /home/mozihao/Dewatermarking/baseline/SLBR/ckpt/pixabay/model_best.pth.tar \
  --use_refine \
  --k_refine ${K_REFINE} \
  --k_skip_stage ${K_SKIP}
  

  
