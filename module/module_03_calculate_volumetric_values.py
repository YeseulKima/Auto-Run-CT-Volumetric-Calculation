import numpy as np
import pandas as pd
import os
import json
import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil

##### Basic utils.
def get_img_arr(img_fPath:str):
    image = sitk.ReadImage(img_fPath)
    img_arr = sitk.GetArrayFromImage(image)
    return img_arr

def is_organ_truncated(mask_fPath):
    mask_img = sitk.ReadImage(mask_fPath)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    first_slice = mask_arr[0, :, :]
    last_slice = mask_arr[mask_arr.shape[0]-1, :, :]

    if np.sum(first_slice) != 0 or np.sum(last_slice) != 0:
        return True
    return False

def save_at_center_L3_img_arr(img_arr, save_fPath:str):
    plt.imshow(img_arr, cmap='gray')  # BL Body Cav Center L3 slice
    plt.show()
    plt.savefig(save_fPath)
    return None

##### "organ_vol_calc_option = 0" / Get volume from the statistics module of TotalSegmentator
def get_roi_volume_from_statistics(path_to_statistics:str, roi_name:str, rounding_up_to=4 ): # Can use it for tissue values
    with open(path_to_statistics, 'r') as file:
        data = json.load(file)
    roi_volume = round(float(data[roi_name]["volume"]), rounding_up_to)
    #organ_intensity = data[organ_name]["intensity"]
    return roi_volume

##### "organ_vol_calc_option = 1" / Get volume from the meta
def calculate_roi_volume_from_metadata(path_to_roi_Nifti_img:str):
    roi_nii_img = sitk.ReadImage(path_to_roi_Nifti_img)
    roi_arr = sitk.GetArrayFromImage(roi_nii_img)

    spacing = roi_nii_img.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    voxel_count = np.sum(roi_arr == 1)
    roi_volume = voxel_count * voxel_volume

    return roi_volume

def find_center_L3_from_SegWh_dirPath(SegWh_dirPath:str):
    bottom_L3, center_L3, top_L3, notes = None, None, None, None
    SegWh_sub_files = os.listdir(SegWh_dirPath)
    if "vertebrae_L3.nii.gz" in SegWh_sub_files:
        L3_nii_fPath = os.path.join(SegWh_dirPath, "vertebrae_L3.nii.gz")
        L3_nii_img = sitk.ReadImage(L3_nii_fPath)
        L3_arr = sitk.GetArrayFromImage(L3_nii_img)
        zs = []
        for x in range(len(L3_arr)):
            if np.sum(L3_arr[x, :, :]) > 0.5:
                zs.append(x)
        if len(zs) > 0:
            bottom_L3 = np.min(zs)
            top_L3 = np.max(zs)
            center_L3 = round((bottom_L3 + top_L3) / 2)
        else:
            print("Goes into L2!!")
            notes = 'L2'
            if "vertebrae_L2.nii.gz" in SegWh_sub_files:
                L3_nii_fPath = os.path.join(SegWh_dirPath, "vertebrae_L2.nii.gz")
                L3_nii_img = sitk.ReadImage(L3_nii_fPath)
                L3_arr = sitk.GetArrayFromImage(L3_nii_img)
                zs = []
                for x in range(len(L3_arr)):
                    if np.sum(L3_arr[x, :, :]) > 0.5:
                        zs.append(x)
                if len(zs) > 0:
                    bottom_L3 = np.min(zs)
                    top_L3 = np.max(zs)
                    center_L3 = round((bottom_L3 + top_L3) / 2)
                else:
                    notes = 'L1'
                    print("Goes into L1!!")
                    if "vertebrae_L1.nii.gz" in SegWh_sub_files:
                        L3_nii_fPath = os.path.join(SegWh_dirPath, "vertebrae_L1.nii.gz")
                        L3_nii_img = sitk.ReadImage(L3_nii_fPath)
                        L3_arr = sitk.GetArrayFromImage(L3_nii_img)
                        zs = []
                        for x in range(len(L3_arr)):
                            if np.sum(L3_arr[x, :, :]) > 0.5:
                                zs.append(x)
                        if len(zs) > 0:
                            bottom_L3 = np.min(zs)
                            top_L3 = np.max(zs)
                            center_L3 = round((bottom_L3 + top_L3) / 2)
    return (bottom_L3, center_L3, top_L3, notes)

def get_roi_volume_at_center_L3(path_to_roi_Nifti_img:str, center_L3, save_fPath=None, is_tissue=False):
    roi_nii_img = sitk.ReadImage(path_to_roi_Nifti_img)
    roi_arr = sitk.GetArrayFromImage(roi_nii_img)
    roi_at_center_L3 = roi_arr[center_L3, :, :]
    voxel_counts_at_center_L3 = np.sum(roi_at_center_L3 == 1)

    spacing = roi_nii_img.GetSpacing()
    if not is_tissue:
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
    if is_tissue:
        print('This is tissue, will calculate pixel not voxel.')
        voxel_volume = spacing[0] * spacing[1]

    roi_volume_at_center_L3 = voxel_counts_at_center_L3 * voxel_volume

    if save_fPath is not None:
        save_at_center_L3_img_arr(roi_at_center_L3, save_fPath=save_fPath)

    return roi_volume_at_center_L3

class volumetric_values_df():
    def __init__(self):
        self.tPoints_to_check = None
        self.organs_to_check = None
        self.tissue_comps_to_check = None
        self.read_dirPath = None
        self.save_dirPath = None
        self.organ_vol_calc_option = 0
        self.initialized_values = [self.tPoints_to_check, self.organs_to_check, self.tissue_comps_to_check, self.read_dirPath, self.save_dirPath]
        self.volumetric_values_df = None

    # Check all requirements are placed.
    @property
    def initialized(self):
        return all(value is not None for value in self.initialized_values)

    def set_tPoints_organs_tissue_branches(self, tPoints_to_check:list, organs_to_check:list, tissue_comps_to_check:list):
        self.tPoints_to_check = tPoints_to_check
        self.organs_to_check = organs_to_check
        self.tissue_comps_to_check = tissue_comps_to_check
        self.tPoint_dict = {key: [] for key in self.tPoints_to_check}  # Upper dict.
        print("Branches are set!")

    def set_read_and_save_dirPath(self, read_dirPath:str, save_dirPath:str, save_fName:str):
        self.read_dirPath = read_dirPath
        self.save_dirPath = save_dirPath
        self.save_fName = save_fName
        print("Read and save dirPaths are set!")

    def check_initialized_values(self):
        print(self.initialized_values)

    def refresh_saving_dicts(self):
        organ_volume_dict = {key: [] for key in self.organs_to_check}
        tissue_vals_dict = {key: [] for key in self.tissue_comps_to_check}
        body_cav_dict = {'body_cav_at_L3': [], 'notes_for_L3': []}
        return (organ_volume_dict, tissue_vals_dict, body_cav_dict)

    def run_calculation(self, is_save_final_df = True, is_save_img_at_center_L3=False):
        self.is_save_img_at_center_L3 = is_save_img_at_center_L3
        if self.is_save_img_at_center_L3:
            img_at_center_L3_dirPath = os.path.join(self.save_dirPath, "temp_imgs_at_center_L3")
            if os.path.exists(img_at_center_L3_dirPath):
                shutil.rmtree(img_at_center_L3_dirPath)
            os.mkdir(img_at_center_L3_dirPath)

        if is_save_final_df:
            print("Run and Save the final calculation results. ")
        else:
            print("NOT saving the final calculation results. Check the is_save_final_df value.")

        organ_volume_dict, tissue_vals_dict, body_cav_dict = self.refresh_saving_dicts()

        for tPoint in self.tPoints_to_check:
            self.PT_IDs = os.listdir(self.read_dirPath) # Be careful for "thumbs.db" 파일 지워야 할 수도...
            for PT_ID in self.PT_IDs:
                print('Start! ', PT_ID, '/ ', tPoint)
                validated_dirPath = os.path.join(self.read_dirPath, str(PT_ID), "Validated_BL_and_PreSurg")
                tp_dirPath = os.path.join(validated_dirPath, tPoint + "_CTScans")
                listdirs = os.listdir(tp_dirPath)

                ##### Read the original img
                for dir in listdirs:
                    if dir.startswith("A"):
                        orig_fPath = os.path.join(tp_dirPath, dir)
                        orig_img_arr = get_img_arr(orig_fPath)  # original img arr

                ##### Read the segmented_organ directory first
                for dir in listdirs:
                    if orig_img_arr is None:
                        print("Original CT img is missing!")

                    else:
                        # 2) Get the organ volume.
                        if "SegWh" in dir:
                            SegWh_dirPath = os.path.join(tp_dirPath, dir)
                            listfiles = os.listdir(SegWh_dirPath)
                            statistics_fPath = os.path.join(SegWh_dirPath, "statistics.json")

                            ## Check the organ is truncated or not, and then get the organs' volume.
                            for organ in self.organs_to_check:
                                organ_mask_fPath = os.path.join(SegWh_dirPath, str(organ) + '.nii.gz')
                                organ_truncated = is_organ_truncated(organ_mask_fPath)  # True= organ is truncated / False = organ is completely located inside the CT img.
                                if not organ_truncated:  # With truncated, returns None.
                                    if self.organ_vol_calc_option == 0 and os.path.exists(statistics_fPath):
                                        organ_volume = get_roi_volume_from_statistics(path_to_statistics=statistics_fPath,roi_name=str(organ))
                                        organ_volume = round(organ_volume, 4)
                                        organ_volume_dict[organ].append(organ_volume)
                                    else:
                                        organ_volume = calculate_roi_volume_from_metadata(path_to_roi_Nifti_img=organ_mask_fPath)
                                        organ_volume = round(organ_volume, 4)
                                        organ_volume_dict[organ].append(organ_volume)
                                else:
                                    organ_volume = None
                                    organ_volume_dict[organ].append(organ_volume)

                            ## Get the center of L3 (or L2 or L1) for the tissue composition values.
                            _, center_L3, _, note_for_L3 = find_center_L3_from_SegWh_dirPath(SegWh_dirPath)
                            body_cav_dict['notes_for_L3'].append(note_for_L3)

                ##### Read the body cavity values.
                for dir in listdirs:
                    if center_L3 is None:
                        print("Location of the center of L3 is missing!")
                    else:
                        if "SegBd" in dir:
                            SegBd_BodyCav_fPath = os.path.join(tp_dirPath, dir, "body_trunc.nii.gz")
                            if self.is_save_img_at_center_L3:
                                img_at_center_L3_fPath = os.path.join(img_at_center_L3_dirPath, str(PT_ID)+"_"+str(tPoint)+"_body_trunc.png")
                            else:
                                img_at_center_L3_fPath = None
                            area_body_cav_at_center_L3 = get_roi_volume_at_center_L3(SegBd_BodyCav_fPath, center_L3, save_fPath=img_at_center_L3_fPath)
                            body_cav_dict['body_cav_at_L3'].append(area_body_cav_at_center_L3)

                ##### Read the body cavity values.
                for dir in listdirs:
                    if "SegTs" in dir:
                        SegTs_dirPath = os.path.join(tp_dirPath, dir)
                        for tissue in self.tissue_comps_to_check:
                            tissue_mask_fPath = os.path.join(SegTs_dirPath, str(tissue) + '.nii.gz')
                            if self.is_save_img_at_center_L3:
                                img_at_center_L3_fPath = os.path.join(img_at_center_L3_dirPath, str(PT_ID)+"_"+str(tPoint)+"_"+str(tissue)+".png")
                            else:
                                img_at_center_L3_fPath = None
                            area_tissue_mask_at_center_L3 = get_roi_volume_at_center_L3(tissue_mask_fPath, center_L3, save_fPath=img_at_center_L3_fPath, is_tissue=True)
                            tissue_vals_dict[tissue].append(area_tissue_mask_at_center_L3)

            values_dict = {**organ_volume_dict, **tissue_vals_dict, **body_cav_dict}
            self.tPoint_dict[tPoint].append(values_dict)

            organ_volume_dict, tissue_vals_dict, body_cav_dict = self.refresh_saving_dicts()  # Refesh the dictionarys

        tPoint_df = pd.DataFrame({'PT_ID': self.PT_IDs})
        for tPoint in self.tPoints_to_check:
            self.tPoint_dict[tPoint] = {str(tPoint) + '_' + key: value for key, value, in self.tPoint_dict[tPoint][0].items()}
            tPoint_df_ = pd.DataFrame(self.tPoint_dict[tPoint])
            tPoint_df = pd.concat([tPoint_df, tPoint_df_], axis=1)
        self.volumetric_values_df = tPoint_df

        if is_save_final_df:
            self.volumetric_values_df.to_csv(os.path.join(self.save_dirPath, self.save_fName), index=False, index_label=False)
        else:
            print("Did not save the final calculation results. Check the is_save_final_df value.")



if __name__ == "__main__":

    ### Control variables.
    tPoints_to_check = ['BL', 'PreSurg']
    organ_vol_calc_option = 1 # 0: Auto / 1: Manually from the metadata.
    organs_to_check = ['liver', 'pancreas']
    tissue_comps_to_check = ['subcutaneous_fat', 'torso_fat', 'skeletal_muscle']
    read_dirPath = "/rsrch5/home/radphys_rsch/ykim23/Pancreatic_Cancer/Converted_Nifti/sorted_pts_with_BL_and_PreSurg_fin/temp"
    save_dirPath = "/rsrch5/home/radphys_rsch/ykim23/Pancreatic_Cancer/Converted_Nifti/sorted_pts_with_BL_and_PreSurg_fin"
    save_fName = "temp_volumetric_vals_df.csv"
    is_save_img_at_center_L3 = True # True: Will save img arrays at center L3.


    vol_df = volumetric_values_df()
    vol_df.set_tPoints_organs_tissue_branches(tPoints_to_check=tPoints_to_check, organs_to_check=organs_to_check, tissue_comps_to_check=tissue_comps_to_check)
    vol_df.set_read_and_save_dirPath(read_dirPath=read_dirPath, save_dirPath=save_dirPath, save_fName=save_fName)
    vol_df.run_calculation(is_save_img_at_center_L3=is_save_img_at_center_L3)















