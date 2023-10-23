# Rerun change detection code
2023.6.13: major revision

## Dataset
beijing: `E:\yinxcao\weaksup\BAdata`
shanghai: `E:\yinxcao\weaksup\BAdata_sh`  
To generate training or test lists, run
```commandline
python zy3bacd_dataset.py
```

## Training 
analysis unit: pixel, object, pixel+object

```commandline
python train_mitb1_0.6_cam_stride_objpix.py
python ttest_mitb1_0.6_cam_stride_objpix.py
```

