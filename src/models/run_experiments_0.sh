########################################################################################################################
########################################################################################################################
seed=0
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.00001\
                  --gpu_id 0 \
                  --arch Xception\
                  --sampling residual_bins\
                  --s_dset_txt "Gravidity"

