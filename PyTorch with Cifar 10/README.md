example code

python main.py --lr=0.01 --layer_chunk=5 --bits=4 --fn "attempt"

lr is learning rate
layer_chunk is from 0-5 (0 for conv1, 5 for FC)
bits = weight precision
--fn is filename, don't need to put .csv
