export CUDA_VISIBLE_DEVICES=0

for model in FEDformer Autoformer Informer Transformer
do

for preLen in 96 192 336 720
do

python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 1  # edited

done
done
