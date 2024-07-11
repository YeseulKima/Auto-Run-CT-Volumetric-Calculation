from totalsegmentator.python_api import totalsegmentator
import os
import time
def run_TotalSegmentator_for_tasks(Converted_NIFTI_dirPath_full:str, task_to_run:list, get_statistics, check_time:bool, fName_max_length=30):
    print("Let's run the TotalSegmentator!")
    if check_time:
        start_time = time.time()
        time.sleep(1)

    if os.path.exists(Converted_NIFTI_dirPath_full):
        PT_IDs = os.listdir(Converted_NIFTI_dirPath_full)
        for PT_ID in PT_IDs:
            ##### Read part,
            Val_dirPath_full = os.path.join(Converted_NIFTI_dirPath_full, PT_ID, 'Validated_BL_and_PreSurg')
            tPoint_dirPaths = os.listdir(Val_dirPath_full)
            for tPoint in tPoint_dirPaths: # ex. 'BL_CTScans' or 'PreSurg_CTScans'
                ##### Read part,
                tPoint_dirPath_full = os.path.join(Val_dirPath_full, tPoint)
                nifti_files = os.listdir(tPoint_dirPath_full)

                for nifti_file in nifti_files:
                    nifti_fPath_full = os.path.join(tPoint_dirPath_full, nifti_file)

                    ##### Run the TotalSegmentator here,
                    for task in task_to_run:

                        ### Set the output directory name,
                        if len(nifti_file) > fName_max_length:
                            nifti_file = nifti_file[:fName_max_length]

                        ## For each task,
                        if task == "total":
                            task_output_dirName = "SegWh_" + str(nifti_file)
                        elif task == "tissue_types":
                            task_output_dirName = "SegTs_" + str(nifti_file)
                        elif task == "body":
                            task_output_dirName = "SegBd_" + str(nifti_file)

                        task_output_dirPath = os.path.join(tPoint_dirPath_full, task_output_dirName)
                        if get_statistics:
                            totalsegmentator(nifti_fPath_full, task_output_dirPath, device='gpu', task=task, statistics=True)
                        else:
                            totalsegmentator(nifti_fPath_full, task_output_dirPath, device='gpu', task=task, statistics=False)
    else:
        print('Error: No Converted Nifti directories to run!')

    if check_time:
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    Converted_NIFTI_dirPath = "/rsrch5/home/radphys_rsch/ykim23/Pancreatic_Cancer/Converted_Nifti/sorted_pts_with_BL_and_PreSurg_fin/test_module/"
    Converted_NIFTI_dirName = "Converted_NIFTIs"
    task_to_run = ["total"] #, "tissue_types", "body"
    get_statistics = False
    fName_max_length = 30
    check_time=False

    run_TotalSegmentator_for_tasks(Converted_NIFTI_dirPath_full=Converted_NIFTI_dirPath+Converted_NIFTI_dirName, task_to_run=task_to_run, get_statistics=get_statistics, check_time=check_time, fName_max_length=fName_max_length)