# AIMAS HW1

Q56104076 陳哲緯

# install

```python
pipenv shell      # enter virtual environment
pipenv install    # install needed package
```

## dataset

The dataset should be placed like:

```
├─EKG_seg
│  ├─0_json
│  ├─1000_json
├─EKG_unzip
│  ├─EKG_001-120
│  ├─EKG_121-240
│  ├─EKG_241-360
│  ├─EKG_361-480
│  └─EKG_481-600
├─imageseg
└─output
└─Q1.py
└─Q2.py
└─Q3.py
└─train.py
└─Pipfile
└─Pipfile.lock
```

# Q1

+ split all image under EKG_unzip to 12 parts and save as  **Q1_data.pkl**

```
python Q1.py
```

# Q2

+ read long lead II from **Q1_data.pkl** 
+ counting heart beat by **cross corrrelation**
+ save output as Q2.csv (column = ["image_path" , "heart beat"])
```
python Q2.py
```
![](./demo_picture/Q2-longlead.jpg)
![](./demo_picture/Q2-peak.jpg)

# Q3
+ train Unet model to segment p(red) and qrs(green) 
