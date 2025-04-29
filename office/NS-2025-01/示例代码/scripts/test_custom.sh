K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res


INPUT_SIZE=512
NAME=slbr_v1
TEST_DIR=/home/mozihao/Dewatermarking/Data/RealWorld_Unlabeled_512

CUDA_VISIBLE_DEVICES=6 python3  test_custom.py \
  --name ${NAME} \
  --nets slbr \
  --models slbr \
  --input-size ${INPUT_SIZE} \
  --crop_size ${INPUT_SIZE} \
  --test-batch 4 \
  --evaluate\
  --preprocess resize \
  --no_flip \
  --mask_mode ${MASK_MODE} \
  --k_center ${K_CENTER} \
  --use_refine \
  --k_refine ${K_REFINE} \
  --k_skip_stage ${K_SKIP} \
  --resume /home/mozihao/Dewatermarking/baseline/SLBR/ckpt/slbr_v1/psnr_38_14.tar \
  --test_dir ${TEST_DIR} 
  
