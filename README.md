# <p align="center">Filmach of the Sad</p>
### Getting Started
To train a model you will need a bunch of `.txt` files and tonnes of patience:
```
python3 train.py \
    --epochs=15 \
    --seq_len=20 \
    --tensorboard_dir=logs
```

To generate your masterpieces it will be enough to run the script and pretend that you are typing:
```
python3 sample.py \
    --length=10000 \
    --start_text='Молодой человек, что' \
    --model_name=model.h5
```

### Example
One day on **WTH 3.0** the first `filmach`'s effort results were presented. The offspring is called **Neurofilm** and flaunts [there](https://youtu.be/L3WV0G_zut8).
### <p align="center">Copyright 2018 [lisovskey](https://t.me/lisovskey)</p>