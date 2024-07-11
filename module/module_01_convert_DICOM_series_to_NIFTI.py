import os
import SimpleITK as sitk
import shutil
def check_prerequisities(run_for_BL_and_PreSurg_comp:bool, DICOM_series_dirPath:str):
    if run_for_BL_and_PreSurg_comp:
        is_prerequisities = check_prerequisites_for_BL_and_PreSurg_comparison(DICOM_series_dirPath)
        if is_prerequisities is None:
            return True
        else:
            return is_prerequisities
    else:
        is_prerequisities = check_prerequisities_for_general_use(DICOM_series_dirPath)
        if is_prerequisities is None:
            return True
        else:
            return is_prerequisities

def check_prerequisites_for_BL_and_PreSurg_comparison(DICOM_series_dirPath:str):
    PT_dirPaths = os.listdir(DICOM_series_dirPath)
    PT_dirPaths = [file for file in PT_dirPaths if '.db' not in file]
    for PT_dirPath in PT_dirPaths:
        PT_dirPath_full = os.path.join(DICOM_series_dirPath, PT_dirPath)
        Val_dirPaths = os.listdir(PT_dirPath_full)
        if 'Validated_BL_and_PreSurg' not in Val_dirPaths:
            print('PreRequisitiesError: No Validated_BL_and_PreSurg')
            print('ERROR check at path: ', PT_dirPath_full)
            return False
        else:
            Val_dirPath_full = os.path.join(PT_dirPath_full, 'Validated_BL_and_PreSurg')
            tPoint_dirPths = os.listdir(Val_dirPath_full)
            if 'BL_CTScans' not in tPoint_dirPths:
                print('PreRequisitiesError: No BL_CTScans!')
                print('ERROR check at path: ', Val_dirPath_full)
                return False
            elif 'PreSurg_CTScans' not in tPoint_dirPths:
                print('PreRequisitiesError: No PreSurg_CTScans!')
                print('ERROR check at path: ', Val_dirPath_full)
                return False
            else:
                for tPoint in tPoint_dirPths:
                    tPoint_dirPath_full = os.path.join(Val_dirPath_full, tPoint)
                    CT_Contents = os.listdir(tPoint_dirPath_full)
                    if len(CT_Contents) == 0:
                        print('PreRequisitiesError: No CT Scans at all!')
                        print('ERROR check at path: ', tPoint_dirPath_full)
                        return False
                    else:
                        for CT_Dir in CT_Contents:
                            CT_Contents_dirPath_full = os.path.join(tPoint_dirPath_full, CT_Dir)
                            DICOM_series = os.listdir(CT_Contents_dirPath_full)
                            for DICOM_file in DICOM_series:
                                if DICOM_file.endswith('.dcm'):
                                    pass
                            if len(DICOM_series)==0:
                                print('PreRequisitiesError: Empty DICOM series!')
                                print('ERROR check at path: ', CT_Contents_dirPath_full)
                                return False

def check_prerequisities_for_general_use(DICOM_series_dirPath:str):
    print('General Use is not yet implemeted!')
    return False

def get_sitk_from_dicom_imgs(file_path: str, is_sample_run=False):
    if is_sample_run:
        dirPath = "../../database/raw_data/sample_data/"
        load_Path = dirPath + file_path
    else:
        load_Path = file_path

    # Load the DICOM image"s"
    reader = sitk.ImageSeriesReader()
    reader.LoadPrivateTagsOn()

    filenamesDICOM = reader.GetGDCMSeriesFileNames(f'{load_Path}')
    reader.SetFileNames(fileNames=filenamesDICOM)
    sitk_orig_DICOM = reader.Execute()

    return sitk_orig_DICOM

def dicom_2_nifti(r_dicom_series_dirPath:str, DICOM_dirName:str, w_nifti_dirPath:str):
    sitk_img_i = get_sitk_from_dicom_imgs(file_path=r_dicom_series_dirPath, is_sample_run=False)  # Get DICOM series.

    save_niiName = DICOM_dirName + ".nii.gz"
    save_nii_fPath = os.path.join(w_nifti_dirPath, save_niiName)
    sitk.WriteImage(sitk_img_i, fileName=save_nii_fPath)
    print("Done making! = ", save_nii_fPath)
    return 0

def if_exist_remove_not_create(dirPath:str):
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
    else:
        shutil.rmtree(dirPath)
def convert_all_PTs(is_prerequisities:bool, DICOM_series_dirPath:str, Converted_NIFTI_dirPath:str, clear_pre_existing_Converted_Nifti_dirPath:bool):
    if is_prerequisities:
        Converted_NIFTI_dirPath_full = os.path.join(Converted_NIFTI_dirPath, "Converted_NIFTIs")
        if os.path.exists(Converted_NIFTI_dirPath_full):
            if clear_pre_existing_Converted_Nifti_dirPath:
                print("Will clear pre-existing Converted_NIFTI_dirPath; ",Converted_NIFTI_dirPath_full )
                shutil.rmtree(Converted_NIFTI_dirPath_full)
                os.mkdir(Converted_NIFTI_dirPath_full)
            else:
                print("Error: Can't clear pre-existing Converted_NIFTI_dirPath; " ,Converted_NIFTI_dirPath_full)
                print("Clear and re-run!")
                pass
        else:
            os.mkdir(Converted_NIFTI_dirPath_full)

        PT_IDs = os.listdir(DICOM_series_dirPath)
        PT_IDs = [file for file in PT_IDs if '.db' not in file]
        for PT_ID in PT_IDs:
            ##### Read part,
            Val_dirPath_full = os.path.join(DICOM_series_dirPath, PT_ID, 'Validated_BL_and_PreSurg')
            tPoint_dirPaths = os.listdir(Val_dirPath_full)

            ##### Write part,
            w_PT_dirPath_full = os.path.join(Converted_NIFTI_dirPath_full, PT_ID)
            if_exist_remove_not_create(w_PT_dirPath_full) # If PT_dirPath not exist, create one. If already exist, remove all trees.
            w_Val_dirPath_full = os.path.join(w_PT_dirPath_full, 'Validated_BL_and_PreSurg')
            if_exist_remove_not_create(w_Val_dirPath_full)

            for tPoint in tPoint_dirPaths: # ex. 'BL_CTScans' or 'PreSurg_CTScans'
                ##### Read part,
                tPoint_dirPath_full = os.path.join(Val_dirPath_full, tPoint)
                DICOM_dirPaths = os.listdir(tPoint_dirPath_full)

                ##### Write part,
                w_tPoint_dirPath_full = os.path.join(w_Val_dirPath_full, tPoint)
                if_exist_remove_not_create(w_tPoint_dirPath_full)

                for DICOM_dirName in DICOM_dirPaths:
                    ##### Read part,
                    DICOM_dirPath_full = os.path.join(tPoint_dirPath_full, DICOM_dirName)

                    ##### Write part,
                    dicom_2_nifti(r_dicom_series_dirPath=DICOM_dirPath_full, DICOM_dirName=DICOM_dirName, w_nifti_dirPath=w_tPoint_dirPath_full)



    else:
        print('Check the pre-requisities!')

if __name__ == "__main__":
    DICOM_series_dirPath = "/rsrch5/home/radphys_rsch/ykim23/Pancreatic_Cancer/Converted_Nifti/sorted_pts_with_BL_and_PreSurg_fin/test_module/DICOM_series_temp"
    run_for_BL_and_PreSurg_comp=True
    Converted_NIFTI_dirPath = "/rsrch5/home/radphys_rsch/ykim23/Pancreatic_Cancer/Converted_Nifti/sorted_pts_with_BL_and_PreSurg_fin/test_module/"

    is_prerequisities = check_prerequisities(run_for_BL_and_PreSurg_comp=run_for_BL_and_PreSurg_comp, DICOM_series_dirPath=DICOM_series_dirPath)
    convert_all_PTs(is_prerequisities=is_prerequisities, DICOM_series_dirPath=DICOM_series_dirPath, Converted_NIFTI_dirPath=Converted_NIFTI_dirPath)

