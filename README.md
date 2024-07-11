# Auto-Run-CT-Volumetric-Calculation

A command line based automatic CT volumetric values calculation tool. 

## Paper
For more details, please see our paper (to/be/updated) which has been accepted at XXXX on YYYY. 
If this code is useful for your work, please consider to cite our paper:
```
@inproceedings{
    kimXXXX,
    title={YYYY},
    author={ZZZZ},
    booktitle={AAAA},
    year={BBBB},
    url={CCCC}
}
```


## Requirements
- randomForestSRC
- caret
- survival
- mlr
- ggRandomForests
- pec
- pROC
- tidyverse


## Description

1) ./randomsurvivalforest_backward_feature_selection/
* RSF_BFS_sample_training.R : A backward feature selection code that uses permutation feature importance as a criterion for selecting features for a random survival forest(RSF) model.
* dummy_pt_data.csv : A dummy patient data set with dummy values for the entire input clinical factors.
* utils/RF_performance_metrix.R : Customized functions to compute performance metrics; Harell's concordance index (C-index) and integrated brier score (IBS).
* utils/rsf_hyperparameter_tuning.R : Customized functions to perform multi-criteria (C-index and IBS) hyper parameter tuning.

2) ./webapp/
* gbm_calculator_codes.R : A Rshiny code to develop our web application.
* finalmodels/OS & finalmodels/PFS : The final 100 cross-validated RSF model for predicting overall survival (OS) and progression-free survival (PFS).

## Authors
  - [YeseulKima](https://github.com/YeseulKima) - **Yeseul Kim** - <yeseulkim@catholic.ac.kr>
  - [wonmo](https://github.com/wonmo) - **Wonmo Sung** - <wsung@catholic.ac.kr>
