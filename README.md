# Auto-Run-CT-Volumetric-Calculation

A command line based automatic CT volumetric values calculation tool. 



## Requirements
Before run the code, install packages from the requirements.txt file. 
- TotalSegmentator >= 2.2
- Python >= 3.9


## Description

### 1) Pre-requisities and notes for warnings
- Recommended input DICOM series should be structurized like the below (click the arrow).  
  ("path_to_input_directory/Patient_Identifier_Number/Validated_BL_and_PreSurg/BL_CTScans(PreSurg_CTScans)/DICOM_series_folder/1.dcm ~ n.dcm")

  <details>
    <summary>Click to see the directory structure</summary>
    
    ![DICOM Series Directory Structure]![image](https://github.com/user-attachments/assets/8f6ad81e-9a34-4a30-ac33-b5679ab99253)
    
  </details>

- Input arguments for module 2 follow that of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator). 
- Don't make the "DICOM_series_folder" empty.
- To run "tissue_types" task, register your own license of TotalSegmentator to the system.

### 2) Notes for the input and the outputs
- **Input**: DICOM series with two different time points; base line (BL) and pre-surgery (PreSurg).
- **Output** for each modules:
  Output from each module becomes the input of the following module.
  - **Module 01**: NIFTI file for each time point.
  - **Module 02**: Auto segmented masks using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) with the given task.
  - **Module 03**: Calculated organ (ex. liver, pancreas) volume in mm3 or tissue (ex. subcutaneous fat, torso fat, skeleton muscle) volume percentage with the body cavity at center of L3. 
  - **Module 04**: Statistical analysis results with given endpoints across groups. 
- Notes for the **"config.json"** file:
  - start_point: 0 will start from the module 01, 1 will from module 02, and so on.
  - run_following_modules: "true" will run all the following modules, "false" will only run the specified module.
  - clear_pre_existing_Converted_Nifti_dirPath: "true" will remove all files in the "Converted_NIFTI_dirPath".
  - get_statistics: "true" will calculate the statistics from the TotalSegmentator module.
  - is_save_img_at_center_L3: "true" will save the axial slice of center of the L3 location.
  - is_save_boxplot: "true" will save the box plots and line plots from the statistical analysis.
    
### 3) Usage
- Check the input arguments from the config file.
- Run the main file like below, and check the outputs.  
     > $python main.py


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
