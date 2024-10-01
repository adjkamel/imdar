This repository contains code for the paper:  [Spatio-temporal invariant descriptors for
skeleton-Based human action recognition](https://) 

## Environment

* Python 3.8.2
* PyTorch 3.9.16
* CUDA 12.0

## Data



### NTU RGB+D 60 / NTU RGB+D 120

Request dataset here: [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

Download skeleton-only datasets (skeleton sequences are in txt format inside the zip file): 
* NTU RGB+D 60 &nbsp;&nbsp;&nbsp;:   nturgbd_skeletons_s001_to_s017.zip
* NTU RGB+D 120&nbsp;&nbsp;:  nturgbd_skeletons_s001_to_s017.zip (NTU RGB+D 60) + nturgbd_skeletons_s018_to_s032.zip

Download missing skeletons lookup files from the [authors' GitHub repo:](https://github.com/shahroudy/NTURGB-D?tab=readme-ov-file#samples-with-missing-skeletons)

* NTU_RGBD_samples_with_missing_skeletons

* NTU_RGBD120_samples_with_missing_skeletons

To generate the .npy skeleton files, put downloaded data and missing skeleton files into the following directory structure, which will contain the resulting for the .npy files as well. The name of directories which will contain the discrtors images under ./data/descriptors/ will have the name format as: 'Dataset_Descripotor_TrainTest_EvalProtocol' where 

Dataset :  ntu60 / ntu120 / ucla 

Descriptors: GD / ADD / JDD/ LAD/ FDD 

TrainTest: train / test 

Protocol: csub / cview / csetup 



```
- data/
  -ntu/
    - txt60skel/
    - txt120skel/
    - npy60skel/
    - npy120skel/
    - NTU_RGBD_samples_with_missing_skeletons.txt
    - NTU_RGBD120_samples_with_missing_skeletons.txt
  -descriptors/
    -ntu60_GD_train_csub/
    -ntu60_JDD_train_csub/
    ... 
    -ntu60_GD_test_csub/
    -ntu60_JDD_test_csub/ 
    ...
    -ntu60_GD_train_cview/
    -ntu60_JDD_train_cview/
    ... 
    -ntu60_GD_test_cview/
    -ntu60_JDD_test_cview/
    ...
    -ntu120_GD_train_csub/
    -ntu120_JDD_train_csub/
    ...
    -ntu120_GD_train_csetup/
    -ntu120_JDD_train_csetup/
    ...
    -ntu120_FDD_test_csub/
    -ntu120_JDD_test_csub/
    ...
    -ntu120_GD_test_csetup/
    -ntu120_JDD_test_csetup/
    ...
```

Insude txt2npy.py (file from [authors' GitHub repo](https://github.com/shahroudy/NTURGB-D?tab=readme-ov-file#samples-with-missing-skeletons)), change load_txt_path, save_npy_path, and missing_file_path as above directories, depending on the dataset (NTU RGB+60 or NTU RGB+120), then run:
```
python txt2npy.py 
```

### NW-CLA

Downlaod dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?e=1&dl=0), then move all_sqe files to ./data/ucla/

For NW-UCLA descriptors' folder names will have the form of 'Dataset_Descripotor_TrainTest' without evaluation protocol since it has only the cross view protocol.

```
 - data/
  -ucla/
  -descriptors/
    -ucla_GD_train/ 
    -ucla_JDD_train/
    ...        
    -ucla_GD_test/
    -ucla_JDD_teste/
    ...
```
## Descriptors generation

Generate the descriptors images (GD, JDD, ADD, LAD, FDD) from the .npy files for each dataset, protocol, and for train or test data.

For  NTU RGB+60 dataset run:

```
python skels2desc_ntu.py -d ntu60 -p csub -dspath ./data/ntu/npy60skel/ -despth ./data/descriptors/ -a train
python skels2desc_ntu.py -d ntu60 -p csub -dspath ./data/ntu/npy60skel/ -despth ./data/descriptors/ -a test
python skels2desc_ntu.py -d ntu60 -p cview -dspath ./data/ntu/npy60skel/ -despth ./data/descriptors/ -a train
python skels2desc_ntu.py -d ntu60 -p cview -dspath ./data/ntu/npy60skel/ -despth ./data/descriptors/ -a test
```

For  NTU RGB+120 dataset run:

```
python skels2desc_ntu.py -d ntu120 -p csub -dspath ./data/npy120skel/ -despth ./data/descriptors/ -a train
python skels2desc_ntu.py -d ntu120 -p csub -dspath ./data/npy120skel/ -despth ./data/descriptors/ -a test
python skels2desc_ntu.py -d ntu120 -p csetup -dspath ./data/npy120skel/ -despth ./data/descriptors/ -a train
python skels2desc_ntu.py -d ntu120 -p csetup -dspath ./data/npy120skel/ -despth ./data/descriptors/ -a test
```

For the NW-UCL, no need to specify the protocol.
```
python skels2desc_ucla.py -d ucla  -dspath ./data/ucla/ -despth ./data/descriptors/ -a train
python skels2desc_ucla.py -d ucla -dspath  ./data/ucla/ -despth ./data/descriptors/ -a test
```

## Training


1. For a specific descriptor, of a specific dataset with a specific evaluation protocol, train its model using:


```
python train.py -d ntu60 -p csub -desc GD -gpu 0
python train.py -d ntu60 -p csub -desc JDD -gpu 0
...
python train.py -d ntu60 -p cview -desc GD -gpu 0
python train.py -d ntu60 -p cview -desc JDD -gpu 0
...
python train.py -d ntu120 -p cview -desc GD -gpu 0
python train.py -d ntu120 -p cview -desc JDD -gpu 0
...
python train.py -d ntu120 -p csetup -desc GD -gpu 0
python train.py -d ntu120 -p csetup -desc JDD -gpu 0
...
python train.py -d ucla -desc GD -gpu 0
python train.py -d ucla -desc JDD -gpu 0
```

The resulting checkpoints models can be found in ./checkpoints/ with the name such as ntu60_csub_GD_1_epoch.tar,  ntu120_csub_FDD_10_epoch.tar, ucla_GD_5_epoch.tar, etc.

2. For a specific dataset with a specific evaluation protocol, train its fusion model using all the descriptor together by running:

```
python train.py -d ntu60 -p csub -desc all -gpu 0
python train.py -d ntu60 -p cview -desc all -gpu 0

python train.py -d ntu120 -p csub -desc all -gpu 0
python train.py -d ntu120 -p csetup -desc all -gpu 0

python train.py -d ucla -desc all -gpu 0
```

## Testing

1.To test the fusion model using the pretrained models for a specific dataset and protocole, move the provided pretrained [models](https://drive.google.com/drive/folders/1AwhMdcxk3TSrlgO0ZiNbj3yOOwte6ncl?usp=drive_link) to ./checkponits/ directory and run: 


```
python test.py -d ntu60 -p csub -desc all -gpu 0
python test.py -d ntu60 -p cview -desc all -gpu 0

python test.py -d ntu120 -p csub -desc all -gpu 0
python test.py -d ntu120 -p csetup -desc all -gpu 0

python test.py -d ucla -desc all -gpu 0
```

##### samples of the  descriptors can be found in ./data/samples/

