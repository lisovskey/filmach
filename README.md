# <p align="center">Filmach of the Sad</p>
### Getting Started
To train a model you will need a bunch of `.txt` files and tonnes of patience:
```
python3 train.py \
    --epochs=10 \
    --seq_len=16 \
    --tensorboard_dir=logs
```

To generate your masterpieces it will be enough to run the script and pretend that you are typing:
```
python3 sample.py \
    --length=10000 \
    --start_text='Молодой человек ' \
    --seq_len=16 \
    --model_name=model.h5
```
### <p align="center">Copyright 2018 [lisovskey](https://t.me/lisovskey)</p>