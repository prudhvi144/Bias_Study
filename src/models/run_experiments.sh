########################################################################################################################
########################################################################################################################
seed=0
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.01\
                  --gpu_id 1 \
                  --arch Xception\
                  --sampling balanced\
                  --s_dset_txt "Sperm Source Race"

python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.001\
                  --gpu_id 1 \
                  --arch Xception\
                  --sampling balanced\
                  --s_dset_txt "Sperm Source Race"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.0001\
                  --gpu_id 1 \
                  --arch Xception\
                  --sampling balanced\
                  --s_dset_txt "Sperm Source Race"