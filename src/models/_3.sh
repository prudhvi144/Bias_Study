########################################################################################################################
########################################################################################################################
seed=0
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.01\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "MaxFSH"

python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "MaxFSH"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.0001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "MaxFSH"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.01\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "AMHLastVal"

python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "AMHLastVal"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.0001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "AMHLastVal"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.01\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "#Oocytes Retrieved"

python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "#Oocytes Retrievede"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.0001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "#Oocytes Retrieved"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.01\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "#Mature Oocytes"

python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "#Mature Oocytes"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.0001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "#Mature Oocytes"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.01\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "#Normal Fertilization (2PN)"

python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "#Normal Fertilization (2PN)"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.0001\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "#Normal Fertilization (2PN)"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.01\
                  --gpu_id 3 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "Well #"

python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.001\
                  --gpu_id 1 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "Well #"
python -u train_cnn.py --mode train \
                  --seed $seed \
                  --lr 0.0001\
                  --gpu_id 1 \
                  --arch Xception\
                  --sampling balanced_bins\
                  --s_dset_txt "Well #"