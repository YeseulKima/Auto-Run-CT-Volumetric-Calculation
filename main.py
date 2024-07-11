import json
from module.module_01_convert_DICOM_series_to_NIFTI import dicom_2_nifti

config = None
def read_config(config_fPath:str="./config.json"):
    with open(config_fPath, 'r') as file:
        config = json.load(file)

    config = {k: v for k, v in config.items() if not k.startswith('_comment')} # Without comments.
    return config

def module_01(config):
    print("Start --------------------------------  module # 01!")
    clear_pre_existing_Converted_Nifti_dirPath = config.get("clear_pre_existing_Converted_Nifti_dirPath")
    run_for_BL_and_PreSurg_comp = config.get("run_for_BL_and_PreSurg_comp")
    DICOM_series_dirPath = config.get('DICOM_series_dirPath')
    Converted_NIFTI_dirPath = config.get('Converted_NIFTI_dirPath')

    from module.module_01_convert_DICOM_series_to_NIFTI import check_prerequisities, convert_all_PTs
    is_prerequisities = check_prerequisities(run_for_BL_and_PreSurg_comp=run_for_BL_and_PreSurg_comp,
                                             DICOM_series_dirPath=DICOM_series_dirPath)
    convert_all_PTs(is_prerequisities=is_prerequisities, DICOM_series_dirPath=DICOM_series_dirPath,
                    Converted_NIFTI_dirPath=Converted_NIFTI_dirPath, clear_pre_existing_Converted_Nifti_dirPath=clear_pre_existing_Converted_Nifti_dirPath)

def module_02(config):
    print("Start --------------------------------  module # 02!")
    Converted_NIFTI_dirPath = config.get("Converted_NIFTI_dirPath")
    Converted_NIFTI_dirName = config.get("Converted_NIFTI_dirName")
    task_to_run = config.get("task_to_run")
    get_statistics = config.get("get_statistics")
    fName_max_length = config.get("fName_max_length")
    check_time = config.get("check_time")

    from module.module_02_run_TotalSegmentator import run_TotalSegmentator_for_tasks
    run_TotalSegmentator_for_tasks(Converted_NIFTI_dirPath_full=Converted_NIFTI_dirPath + Converted_NIFTI_dirName,
                                   task_to_run=task_to_run, get_statistics=get_statistics, check_time=check_time,
                                   fName_max_length=fName_max_length)

def module_03(config):
    print("Start --------------------------------  module # 03!")
    tPoints_to_check = config.get('tPoints_to_check')
    organs_to_check = config.get('organs_to_check')
    tissue_comps_to_check = config.get('tissue_comps_to_check')
    read_dirPath = config.get('read_dirPath')
    save_dirPath = config.get('save_dirPath')
    save_fName = config.get('save_fName')
    is_save_img_at_center_L3 = config.get('is_save_img_at_center_L3')

    from module.module_03_calculate_volumetric_values import volumetric_values_df
    vol_df = volumetric_values_df()
    vol_df.set_tPoints_organs_tissue_branches(tPoints_to_check=tPoints_to_check, organs_to_check=organs_to_check,
                                              tissue_comps_to_check=tissue_comps_to_check)
    vol_df.set_read_and_save_dirPath(read_dirPath=read_dirPath, save_dirPath=save_dirPath, save_fName=save_fName)
    vol_df.run_calculation(is_save_img_at_center_L3=is_save_img_at_center_L3)

def module_04(config):
    print("Start --------------------------------  module # 04!")
    ##### From module 03
    tPoints_to_check = config.get('tPoints_to_check')
    organs_to_check = config.get('organs_to_check')
    tissue_comps_to_check = config.get('tissue_comps_to_check')
    read_dirPath = config.get('read_dirPath')
    save_dirPath = config.get('save_dirPath')

    ##### From module 04
    load_prepcoessed_df_fName = config.get("load_prepcoessed_df_fName")
    read_scoring_df_fPath = config.get("read_scoring_df_fPath")
    PT_ID_colName_in_scoring_df = config.get("PT_ID_colName_in_scoring_df")
    save_fName_post_proc_df = config.get("save_fName_post_proc_df")
    save_fName_stat_anal_df = config.get("save_fName_stat_anal_df")
    endpoints_to_check = config.get("endpoints_to_check")
    is_save_boxplot = config.get("is_save_boxplot")
    # load_prepcoessed_df_fName 가 start point에 따라서 save_fName과 같아야 한다는 거 룰 세워.

    from module.module_04_do_statistical_analysis import post_processed_volumetric_df

    proc_vol_df = post_processed_volumetric_df()
    proc_vol_df.set_tPoints_organs_tissue_branches(tPoints_to_check=tPoints_to_check,
                                                   organs_to_check=organs_to_check,
                                                   tissue_comps_to_check=tissue_comps_to_check)
    proc_vol_df.set_read_and_save_dirPath(read_dirPath=read_dirPath, save_dirPath=save_dirPath,
                                          read_scoring_df_fPath=read_scoring_df_fPath,
                                          load_prepcoessed_df_fName=load_prepcoessed_df_fName,
                                          save_fName_post_proc_df=save_fName_post_proc_df,
                                          save_fName_stat_anal_df=save_fName_stat_anal_df)
    proc_vol_df.run_post_processing()

    proc_vol_df.initialize_for_stat_analysis(read_scoring_df_fPath=read_scoring_df_fPath,
                                             PT_ID_colName_in_scoring_df=PT_ID_colName_in_scoring_df,
                                             endpoints_to_check=endpoints_to_check, is_save_boxplot=is_save_boxplot)
    proc_vol_df.run_stats()

def main():

    config = read_config()
    #### Load the control variables.
    start_point = config.get('start_point')
    run_following_modules = config.get('run_following_modules')

    if run_following_modules: # True: Run all following modules.
        if start_point <1: ##### Module 01
            module_01(config)
        elif start_point <2: ##### Module 02
            module_02(config)
        elif start_point <3: ##### Module 03
            module_03(config)
        elif start_point <4: ##### Module 04
            module_04(config)
    else:
        if start_point == 0: ##### Module 01
            module_01(config)
        elif start_point == 1: ##### Module 02
            module_02(config)
        elif start_point == 2: ##### Module 03
            module_03(config)
        elif start_point == 3: ##### Module 04
            module_04(config)


if __name__ == "__main__":
    print("Let's swing!")

    main()
