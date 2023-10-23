# BANet code
2023.6.6: major revision

## Dataset
download link in [google drive](https://drive.google.com/drive/folders/1oxTBi8_tWT0EcflZWH71ECRY_K7WMhfo?usp=share_link) or [Baiduyunpan]()
BA dataset excluding test cities (Beijing, Shanghai, Xian, and Kunming)
### Setup datasets:
* training dataset: E:/yinxcao/ZY3LC/datanew8bit/datalist_posneg_train_0.6.csv
* test dataset: E:/yinxcao/ZY3LC/datanew8bit/datalist_posneg_test_0.6.csv

## Part1. Training for BA detection
- training a multi-scale classification network
- generating the multi-scale cam
- generating pseudo-labels with CRF and thresholding
- adaptive online noise correction for BA detection: 1) obtain correction time; 2) correct labels 
```commandline
python train_mitb1_0.6_cam_stride_tlcmulti4.py
python demo_cues_torch_lvwang_mitb1_cam_stride_tlc_multi4.py
python cam_to_ir_label_tlcmult4.py
python train_mitb1_0.6_cam_stride_tlcmulti4_RRM_adele.py
python ttest_mitb1_0.6_cam_stride_rrm_tlcmulti4_adele.py
```

## Part2. Training for BA change detection
- predict BA results for each date
```commandline
python predict_rrm_tlcmulti4_adele_wholeimg.py
```
1. generate pseudo labels at pixel, object, and pixel+object levels
```matlab
demo_1116_gen_pix_beijing.m
demo_1117_gen_obj_diff_beijing.m
demo_1117_gen_change_cert.m
```
clip sample
```matlab
demo_1117_clipsample_imglab_bj.m
demo_1118_testsample_diffarea.m
```
2. training change detection models
see directory `BANetCD`


#### stats model parameters
see package `torchstat`
