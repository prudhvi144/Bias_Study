########################################################################################################################
########################################################################################################################
seed=0
#python -u train_cnn.py --mode train \
#                  --seed $seed \
#                  --lr 0.01\
#                  --gpu_id 0 \
#                  --arch Xception\
#                  --sampling stratfied\
#                  --s_dset_txt "Sperm Source"

python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.001\
                  --gpu_id 2 \
                  --arch Xception\
                  --sampling stratfied\
                  --s_dset_txt "Sperm Source"
#python -u train_cnn.py --mode train \
#                  --seed $seed \
#                  --lr 0.0001\
#                  --gpu_id 2 \
#                  --arch Xception\
#                  --sampling stratfied\
#                  --s_dset_txt "Sperm Source"