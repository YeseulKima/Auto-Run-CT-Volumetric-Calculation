# Auto-Run-CT-Volumetric-Calculation

A command line based automatic CT volumetric values calculation tool. 



## Requirements
Before run the code, install packages from the requirements.txt file. 
- TotalSegmentator >= 2.2
- Python >= 3.9


## Description

1) Pre-requisities and notes for warnings
- Recommended input DICOM series should be structurized like the below (click the arrow).  
  ("path_to_input_directory/Patient_Identifier_Number/Validated_BL_and_PreSurg/BL_CTScans(PreSurg_CTScans)/DICOM_series_folder/1.dcm ~ n.dcm")

  <details>
    <summary>Click to see the directory structure</summary>
    
    ![DICOM Series Directory Structure]![image](https://github.com/user-attachments/assets/8f6ad81e-9a34-4a30-ac33-b5679ab99253)
    
  </details>

- Don't make the "DICOM_series_folder" empty.
- 

2) Usage
- Input: DICOM series with two different time points; base line (BL) and pre-surgery (PreSurg).
- Output for each modules:
  - NIFTI file for each time point (module 01), 

3) Module for the calculation


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

## Authors
  - [YeseulKima](https://github.com/YeseulKima) - **Yeseul Kim** - <YKim23@mdanderson.org>
