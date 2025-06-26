export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 10    --batch_size 256  --mode train --dataset PSM  --data_path /home/tlmsq/Anomaly-Detection-Prediction/dataset/tube1 --input_c 15    --output_c 15