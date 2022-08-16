########################################################################################################################
########################################################################################################################
seed=0
#python -u train_cnn.py --mode train \
#                  --seed $seed \
#                  --lr 0.01\
#                  --gpu_id 2 \
#                  --arch Xception\
#                  --sampling stratfied\
#                  --s_dset_txt "Sperm Quality (1=Great, 4=Poor)"

#python -u train_cnn.py --mode train \
#                  --seed $seed \
#                  --lr 0.001\
#                  --gpu_id 2 \
#                  --arch Xception\
#                  --sampling stratfied\
#                  --s_dset_txt "Insemination Method"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.0001\
                  --gpu_id 2 \
                  --arch Xception\
                  --sampling stratfied\
                  --s_dset_txt "Insemination Method"