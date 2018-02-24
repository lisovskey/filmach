# <p align="center">Filmach of the Sad</p>
### Getting Started
To train a model you will need a bunch of `.txt` files and tonnes of patience:
```
python3 train.py
    --data_dir=films
    --epochs=50
    --batch_size=256
    --seq_len=64
    --tensorboard_dir=logs
```

To generate your masterpieces it will be enough to run the script and pretend that you are typing:
```
python3 sample.py
    --length=10000
    --start_text='ты просто пристал свое подумать с тебя в темноте и старика всего'
    --seq_len=64
    --model_name=model.h5
```
### <p align="center">Copyright 2018 [lisovskey](https://t.me/lisovskey)</p>