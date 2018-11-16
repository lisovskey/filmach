# <p align="center">Filmach of the Sad</p>
### Getting Started
To train a model you will need a bunch of `.txt` files and tonnes of patience:
```
python3 train.py \
    --epochs=16 \
    --sequence_size=16
```

To generate your masterpieces it will be enough to run the script and pretend that you are typing:
```
python3 sample.py \
    --length=10000 \
    --start_text='Молодой человек,' \
    --model_name=model_16
```

### Example
One day on **WTH 3.0** the first `filmach`'s effort results were presented. The offspring is called **Neurofilm** and flaunts [there](https://youtu.be/L3WV0G_zut8).
### <p align="center">Copyright 2018 [lisovskey](https://t.me/lisovskey)</p>