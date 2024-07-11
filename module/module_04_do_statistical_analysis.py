import numpy as np
import pandas as pd
import os
from scipy import stats
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
##### Stat utils.
def get_distribution(data, confidence=0.95):
    mean = np.mean(data)
    median = round(np.median(data),4)

    n = len(data)
    se = stats.sem(data)  # std
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)  # for ci
    median_ci_low = round(median - h, 4)
    median_ci_high = round(median + h, 4)
    return (mean, median, median_ci_low, median_ci_high)

def check_normality(data, alpha=0.05):
    stat, p_value = stats.shapiro(data)

    '''print(f"Shapiro-Wilk test statistics: {stat}")
    print(f"p-value: {p_value}")

    # p-value 해석
    if p_value > alpha:
        print("귀무가설 채택: 데이터가 정규 분포를 따릅니다.")
    else:
        print("귀무가설 기각: 데이터가 정규 분포를 따르지 않습니다.")'''

    return round(p_value, 9)

def run_student_t_test(data_1, data_2):
    # Perform Levene's test
    levene_stat, levene_p_value = stats.levene(data_1, data_2)

    #print(f"Levene's test statistic: {levene_stat}")
    #print(f"p-value: {levene_p_value}")

    # Interpret Levene's test result
    alpha = 0.05
    if levene_p_value > alpha:
        #print("Fail to reject the null hypothesis: The variances of the two datasets are not significantly different. (Equal variances assumed)")
        equal_var = True
    else:
        #print("Reject the null hypothesis: The variances of the two datasets are significantly different. (Equal variances not assumed)")
        equal_var = False

    # Perform t-test (choose based on variance equality)
    t_stat, p_value = stats.ttest_ind(data_1, data_2, equal_var=equal_var)
    return round(p_value, 9)

def run_wilcoxon_rank_sum_test(data_1, data_2):
    # Perform Wilcoxon signed-rank test
    w_stat, p_value = stats.mannwhitneyu(data_1, data_2)
    return round(p_value, 9)


def draw_box_plot_with_Pval(data_1, data_2, p_val, labels: list, save_fPath: str, fig_title: str, y_label: str,
                            endpoint:str, margin_perc=0.1,):
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,  # Increase the base font size by 2 points (default is 10)
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    # Combine data for plotting
    data = [data_1, data_2]

    # Will express all in median.
    '''# Compute 95% CI
    def compute_ci(data):
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = sem * stats.t.ppf((1 + 0.95) / 2., n - 1)
        return mean - ci, mean + ci

    ci_1 = compute_ci(data_1)
    ci_2 = compute_ci(data_2)'''

    _, median_1, median_ci_low_1, median_ci_high_1 = get_distribution(data_1)
    _, median_2, median_ci_low_2, median_ci_high_2 = get_distribution(data_2)
    ci_1 = (median_ci_low_1, median_ci_high_1 )
    ci_2 = (median_ci_low_2, median_ci_high_2)
    ci_h_1 = median_1 - median_ci_low_1
    ci_h_2 = median_2 - median_ci_low_2

    # Determine y-axis limits with margin using second smallest and second largest values
    all_data = np.concatenate((data_1, data_2))
    y_min = np.partition(all_data, 1)[1]  # Second smallest value
    y_max = np.partition(all_data, -2)[-2]  # Second largest value
    margin = abs(y_max - y_min) * margin_perc  # Apply absolute value to the difference
    y_min -= margin
    y_max += margin

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 10))  # Increased height for better space
    sns.boxplot(data=data, ax=ax, palette="Set1")

    sns.stripplot(data=data, jitter=True, color='black', ax=ax)

    # Add single p-value
    def add_pval_annotation(ax, p_value, x_positions, y_position):
        if p_value < 0.05:
            label = '***'
        elif p_value < 0.1:
            label = '**'
        elif p_value < 0.2:
            label = '*'
        else:
            label = ''

        ax.text((x_positions[0] + x_positions[1]) / 2, y_position, f'p={p_value:.3f} {label}',
                ha='center', va='bottom', color='black', fontsize=18)
        ax.plot(x_positions, [y_position - 2] * 2, lw=1.5, color='black')  # Add horizontal line

    # Add CI to the plot
    ax.errorbar(x=0, y=np.median(data_1), yerr=ci_h_1, fmt='o', color='black', capsize=5)
    ax.errorbar(x=1, y=np.median(data_2), yerr=ci_h_2, fmt='o', color='black', capsize=5)

    '''ax.errorbar(x=0, y=np.median(data_1), yerr=[[np.median(data_1) - ci_1[0]], [ci_1[1] - np.median(data_1)]], fmt='o',
                color='black', capsize=5)
    ax.errorbar(x=1, y=np.median(data_2), yerr=[[np.median(data_2) - ci_2[0]], [ci_2[1] - np.median(data_2)]], fmt='o',
                color='black', capsize=5)'''

    # Add p-value annotation at the center top of the plot
    add_pval_annotation(ax, p_val, [0, 1], y_max - margin/2)

    # Customize plot
    ax.set_xticklabels([f'{labels[0]} (N={len(data_1)})', f'{labels[1]} (N={len(data_2)})'])
    ax.set_title(fig_title)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  # Set y-axis format to show up to two decimal places
    plt.ylim(y_min, y_max)  # Set y-axis limits with margin

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.7)

    plt.tight_layout()  # Adjust the layout to make room for the ylabel
    plt.savefig(save_fPath, bbox_inches='tight')  # Save the plot as a file with tight bounding box
    plt.show()

def draw_line_plot_by_group(data_1, data_2, groups, PT_ID_labels: list, save_fPath, fig_title, y_label:str):
    plt.rcParams.update({
        'font.size': 14,  # Increase the base font size by 2 points (default is 10)
        'axes.titlesize': 19,
        'axes.labelsize': 23,
        'xtick.labelsize': 20,
        'ytick.labelsize': 16,
        'legend.fontsize': 14
    })

    fig, ax = plt.subplots(figsize=(8, 14))

    # Color setup
    if 'High Delta' in groups:
        colors =  ['red' if x == 'High Delta' else 'blue' for x in groups]
    elif 'Type I' in groups:
        colors =  ['green' if x == 'Type I' else 'orange' for x in groups]

    for i, color in zip(range(len(data_1)), colors):
        ax.plot([0, 1], [data_1[i], data_2[i]], marker='o', color=color,
                label=f'Patient {PT_ID_labels[i]}', alpha=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'Presurgery'])

    ax.set_ylabel(y_label)
    ax.set_title(fig_title)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    plt.tight_layout()
    plt.savefig(save_fPath)
    plt.close(fig)


def draw_line_plot_by_group_gr_by_gr(data_1, data_2, groups, PT_ID_labels: list, save_fPath, fig_title, y_label:str):

    plt.rcParams.update({
        'font.size': 14,  # Increase the base font size by 2 points (default is 10)
        'axes.titlesize': 19,
        'axes.labelsize': 23,
        'xtick.labelsize': 20,
        'ytick.labelsize': 16,
        'legend.fontsize': 14
    })

    fig, ax = plt.subplots(figsize=(8, 14))

    ##### Add the distribution lines.
    if 'High Delta' in groups:
        group_1_counts = groups.count('High Delta')
        colors = ['red', 'blue']
        if groups[0] == 'High Delta':
            pass
        else:
            print('High Delta should come first')
    elif 'Type I' in groups:
        group_1_counts = groups.count('Type I')
        colors = ['green', 'orange']
        if groups[0] == 'Type I':
            pass
        else:
            print('Type I should come first')

    data_1_group_1 = data_1[:group_1_counts]
    data_1_group_2 = data_1[group_1_counts:]

    data_2_group_1 = data_2[:group_1_counts]
    data_2_group_2 = data_2[group_1_counts:]

    def remove_missing_elements(list1, list2):
        if len(list1) != len(list2):
            raise ValueError("Both lists must have the same length.")

        filtered_list1 = []
        filtered_list2 = []

        for elem1, elem2 in zip(list1, list2):
            if not (elem1 is None or elem2 is None or np.isnan(elem1) or np.isnan(elem2)):
                filtered_list1.append(elem1)
                filtered_list2.append(elem2)

        return filtered_list1, filtered_list2

    data_1_group_1, data_2_group_1 = remove_missing_elements(data_1_group_1, data_2_group_1)
    data_1_group_2, data_2_group_2 = remove_missing_elements(data_1_group_2, data_2_group_2)

    _, median_d1_g1, median_ci_low_d1_g1, median_ci_high_d1_g1 = get_distribution(data_1_group_1)
    _, median_d1_g2, median_ci_low_d1_g2, median_ci_high_d1_g2 = get_distribution(data_1_group_2)
    _, median_d2_g1, median_ci_low_d2_g1, median_ci_high_d2_g1 = get_distribution(data_2_group_1)
    _, median_d2_g2, median_ci_low_d2_g2, median_ci_high_d2_g2 = get_distribution(data_2_group_2)

    ###### Linear regression
    time_points = np.array([0, 1])

    ### Group 1 Linear regression.
    X_group_1 = np.repeat(time_points, len(data_1_group_1)).reshape(-1, 1)
    y_group_1 = np.concatenate([data_1_group_1, data_2_group_1])

    model_group_1 = LinearRegression().fit(X_group_1, y_group_1)
    y_pred_group_1 = model_group_1.predict(X_group_1)
    r2_group_1 = r2_score(y_group_1, y_pred_group_1)

    ### Group 2 Linear regression.
    X_group_2 = np.repeat(time_points, len(data_1_group_2)).reshape(-1, 1)
    y_group_2 = np.concatenate([data_1_group_2, data_2_group_2])

    model_group_2 = LinearRegression().fit(X_group_2, y_group_2)
    y_pred_group_2 = model_group_2.predict(X_group_2)
    r2_group_2 = r2_score(y_group_2, y_pred_group_2)

    p_val_wilcoxon_g1 = round(run_wilcoxon_rank_sum_test(data_1_group_1, data_2_group_1),4)
    p_val_wilcoxon_g2 = round(run_wilcoxon_rank_sum_test(data_1_group_2, data_2_group_2),4)

    label_g1 = 'R2='+str(round(r2_group_1, 4))+' / P='+str(p_val_wilcoxon_g1)
    label_g2 = 'R2='+str(round(r2_group_2, 4))+' / P='+str(p_val_wilcoxon_g2)

    ax.plot([0,1], [median_d1_g1, median_d2_g1], marker='o', color= colors[0], label=label_g1)
    #ax.fill_between([0, 1], [median_ci_low_d1_g1, median_ci_low_d2_g1], [median_ci_high_d1_g1, median_ci_high_d2_g1], color=colors[0], alpha=0.2)
    ax.plot([0,1], [median_d1_g2, median_d2_g2], marker='o', color= colors[1], label=label_g2)
    #ax.fill_between([0, 1], [median_ci_low_d1_g2, median_ci_low_d2_g2], [median_ci_high_d1_g2, median_ci_high_d2_g2], color=colors[1], alpha=0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'Presurgery'])

    ax.legend(loc='best')
    ax.set_ylabel(y_label)
    ax.set_title(fig_title)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    fig.patch.set_alpha(0.0)  # Figure 배경 투명하게 설정
    ax.patch.set_alpha(0.0)  # Axes 배경 투명하게 설정

    plt.tight_layout()
    plt.savefig(save_fPath)
    plt.close(fig)

class post_processed_volumetric_df():
    def __init__(self):
        self.tPoints_to_check = None
        self.organs_to_check = None
        self.tissue_comps_to_check = None
        self.read_dirPath = None
        self.save_dirPath = None
        self.load_prepcoessed_df_fName = None
        self.initialized_values = [self.tPoints_to_check, self.organs_to_check, self.tissue_comps_to_check, self.read_dirPath, self.save_dirPath, self.load_prepcoessed_df_fName]

        # Storing values.
        self.volumetric_values_dict = {}

    def set_tPoints_organs_tissue_branches(self, tPoints_to_check: list, organs_to_check: list, tissue_comps_to_check: list):
        self.tPoints_to_check = tPoints_to_check
        self.organs_to_check = organs_to_check
        self.tissue_comps_to_check = tissue_comps_to_check
        self.tPoint_dict = {key: [] for key in self.tPoints_to_check}  # Upper dict.
        print("Branches are set!")

    def set_read_and_save_dirPath(self, read_dirPath, save_dirPath, read_scoring_df_fPath, load_prepcoessed_df_fName, save_fName_post_proc_df, save_fName_stat_anal_df):
        self.read_dirPath = read_dirPath
        self.save_dirPath = save_dirPath
        self.read_scoring_df_fPath = read_scoring_df_fPath
        self.load_prepcoessed_df_fName = load_prepcoessed_df_fName
        self.save_fName_post_proc_df = save_fName_post_proc_df
        self.save_fName_stat_anal_df = save_fName_stat_anal_df
        print("Read and save dirPaths are set!")


    # Check all requirements are placed.
    @property
    def initialized(self):
        return all(value is not None for value in self.initialized_values)

    def read_orig_volumetric_df(self):
        orig_volumetric_df_fPath = os.path.join(self.save_dirPath, self.load_prepcoessed_df_fName)
        orig_volumetric_df = pd.read_csv(orig_volumetric_df_fPath)
        return orig_volumetric_df

    def load_volumetric_vals_dict_at_tPoint(self, orig_volumetric_df, tPoint:str):
        volumetric_vals = self.organs_to_check+self.tissue_comps_to_check + ['body_cav_at_L3']
        cols_tPoint = [x+'_'+y for x, y in zip(np.repeat(tPoint, len(volumetric_vals)), volumetric_vals)]
        orig_volumetric_vals = [orig_volumetric_df[col].tolist() for col in cols_tPoint]

        volumetric_vals_dict = {key: values for key, values in zip(cols_tPoint, orig_volumetric_vals)}
        return volumetric_vals_dict

    def tissue_comp_convert_to_vol_percentage(self, tissue_comp_value:float, body_cav_value:float):
        return round(tissue_comp_value/body_cav_value*100, 4)

    def calculate_the_diff_btw_two_tPoints(self, tPoint_BL_roi_vals, tPoint_PreSurg_roi_vals):
        return [round(y-x, 4) for x, y in zip(tPoint_BL_roi_vals, tPoint_PreSurg_roi_vals)]

    def run_post_processing(self):
        orig_volumetric_df = pd.read_csv(os.path.join(self.save_dirPath, self.load_prepcoessed_df_fName))
        self.volumetric_values_dict['PT_ID'] = orig_volumetric_df['PT_ID'].tolist()

        # Tissue compo values convert into volume percentage.
        for tPoint in self.tPoints_to_check:
            volumetric_vals_dict = self.load_volumetric_vals_dict_at_tPoint(orig_volumetric_df=orig_volumetric_df, tPoint=tPoint)
            body_cav_values = volumetric_vals_dict[tPoint+'_body_cav_at_L3']
            for tissue in self.tissue_comps_to_check:
                tissue_values = volumetric_vals_dict[tPoint + '_'+tissue]
                tissue_comp_in_vol_perc = [self.tissue_comp_convert_to_vol_percentage(x, y) for x, y in zip(tissue_values, body_cav_values)]
                volumetric_vals_dict[tPoint + '_'+tissue+'_in_volPerc'] = tissue_comp_in_vol_perc
            self.volumetric_values_dict = self.volumetric_values_dict | volumetric_vals_dict

        # Calculate the differences btw BL and PreSurg time point.
        self.Diff_cols = []
        for organ in self.organs_to_check:
            tPoint_BL_roi_vals = self.volumetric_values_dict['BL_'+organ] # For now, will just put it as string. not arguments.
            tPoint_PreSurg_roi_vals = self.volumetric_values_dict['PreSurg_'+organ]
            roi_diff_btw_two_tPoints = self.calculate_the_diff_btw_two_tPoints(tPoint_BL_roi_vals=tPoint_BL_roi_vals, tPoint_PreSurg_roi_vals=tPoint_PreSurg_roi_vals)
            self.volumetric_values_dict['Diff_'+organ] = roi_diff_btw_two_tPoints
            self.Diff_cols.append('Diff_'+organ)

        for tissue in self.tissue_comps_to_check:
            tPoint_BL_roi_vals = self.volumetric_values_dict['BL_' + tissue + '_in_volPerc']  # For now, will just put it as string. not arguments.
            tPoint_PreSurg_roi_vals = self.volumetric_values_dict['PreSurg_' + tissue + '_in_volPerc']
            roi_diff_btw_two_tPoints = self.calculate_the_diff_btw_two_tPoints(tPoint_BL_roi_vals=tPoint_BL_roi_vals, tPoint_PreSurg_roi_vals=tPoint_PreSurg_roi_vals)
            self.volumetric_values_dict['Diff_' + tissue + '_in_volPerc'] = roi_diff_btw_two_tPoints
            self.Diff_cols.append('Diff_' + tissue + '_in_volPerc')

        volumetric_values_df = pd.DataFrame(self.volumetric_values_dict)
        volumetric_values_df.to_csv(os.path.join(self.save_dirPath, self.save_fName_post_proc_df), index=False, index_label=False)
    ##### Stat.
    def initialize_for_stat_analysis(self, read_scoring_df_fPath, PT_ID_colName_in_scoring_df, endpoints_to_check, is_save_boxplot):
        self.scoring_df = pd.read_csv(read_scoring_df_fPath)
        self.PT_ID_colName_in_scoring_df = PT_ID_colName_in_scoring_df
        self.endpoints_to_check = endpoints_to_check
        self.stat_summary_df = {} # Create an empty dict.
        self.is_save_boxplot = is_save_boxplot

    def split_into_groups_by_endpoint(self, endpoint:str): #volumetric_values_dict['PT_ID']
        categories = self.endpoints_to_check[endpoint] #ex. ['High Delta', 'Low Delta']
        PT_ID_by_category = {}
        for category in categories:
            PT_IDs = self.scoring_df[self.scoring_df[endpoint]==category][self.PT_ID_colName_in_scoring_df]
            PT_ID_by_category[category] = PT_IDs
        return PT_ID_by_category

    def load_volumetric_vals_for_selected_PT_IDs(self, volumetric_values_df, selected_PT_IDs: list):
        selected_df = volumetric_values_df[volumetric_values_df['PT_ID'].isin(selected_PT_IDs)]
        return selected_df

    def run_stats(self):
        # Load data frame.
        self.volumetric_values_df = pd.DataFrame(self.volumetric_values_dict)

        for endpoint in self.endpoints_to_check:
            stat_summary_dict = {}
            rois = self.organs_to_check + self.tissue_comps_to_check
            stat_summary_dict['ROI'] = rois

            volumetric_vals_temp_dict = {}
            PT_ID_by_category = self.split_into_groups_by_endpoint(endpoint=endpoint) # PT_ID by groups.

            ##### 1) To make the stat summary table.
            for category in self.endpoints_to_check[endpoint]:
                volumetric_values_df_by_category = self.volumetric_values_df[self.volumetric_values_df['PT_ID'].isin(PT_ID_by_category[category])]
                volumetric_vals_temp_dict[category] = volumetric_values_df_by_category

                ### Get the distribution values.
                ranges = []
                normality_pvals = []
                for Diff_col_name in self.Diff_cols:
                    data_roi = volumetric_values_df_by_category[Diff_col_name].tolist()
                    data_roi = [x for x in data_roi if not math.isnan(x)]
                    if len(data_roi) > 1:
                        _, median, median_ci_low, median_ci_high = get_distribution(data_roi)
                        ranges.append(str(median) + " [" + str(median_ci_low) + " - " + str(median_ci_high) + "]")
                        normality_p_value = check_normality(data_roi)
                        normality_pvals.append(normality_p_value)
                    else:
                        ranges.append(None)
                        normality_pvals.append(None)

                stat_summary_dict['Range_'+str(category)] = ranges
                stat_summary_dict['Normality_'+str(category)] = normality_pvals

            normality_pvalues_pair = [stat_summary_dict['Normality_' + str(category)] for category in self.endpoints_to_check[endpoint]]
            which_test = ['student' if x > 0.05 and y > 0.05 else 'wilcoxon' for x, y in zip(normality_pvalues_pair[0], normality_pvalues_pair[1])]
            stat_summary_dict['which_test'] = which_test

            test_pvals = []
            for Diff_col_name_i in range(len(self.Diff_cols)):
                Diff_col_name = self.Diff_cols[Diff_col_name_i]
                datas = [volumetric_vals_temp_dict[category][Diff_col_name].tolist() for category in self.endpoints_to_check[endpoint]] # Always High Delta, Type I first.
                data_1 = [x for x in datas[0] if not math.isnan(x)]
                data_2 = [x for x in datas[1] if not math.isnan(x)]
                if which_test[Diff_col_name_i] == 'student':
                    p_value = run_student_t_test(data_1, data_2)
                elif which_test[Diff_col_name_i] == 'wilcoxon':
                    p_value = run_wilcoxon_rank_sum_test(data_1, data_2)
                test_pvals.append(p_value)

                ##### 2) To make the stat summary box plots.
                if self.is_save_boxplot:
                    boxplot_dirPath = os.path.join(self.save_dirPath, "stat_boxplots")
                    if not os.path.exists(boxplot_dirPath):
                        os.makedirs(boxplot_dirPath)
                    fName = str(endpoint)+"_"+str(Diff_col_name)+"_boxplot.png"
                    fig_title = 'Data 1 vs Data 2 with 95% CI and p-value'

                    if Diff_col_name_i <2:
                        y_label = 'mm3'
                    else:
                        y_label = 'Vol%'
                    draw_box_plot_with_Pval(data_1, data_2, p_value, labels=self.endpoints_to_check[endpoint], save_fPath=os.path.join(boxplot_dirPath, fName),
                                            fig_title=fig_title, y_label=y_label, margin_perc = 0.1, endpoint=endpoint)

            stat_summary_dict['p_value'] = test_pvals

            stat_summary_df = pd.DataFrame(stat_summary_dict)
            fName = str(endpoint) + self.save_fName_stat_anal_df
            stat_summary_df.to_csv(os.path.join(self.save_dirPath, fName), index=False, index_label=False)

            ##### 3) To get the line plot by groups.
            if self.is_save_boxplot:
                #print(self.volumetric_values_df)
                vol_vals_dict_by_groups = {'BL':[], 'PreSurg':[], 'groups':[], 'PT_IDs':[], 'roi':[]}
                group_keys = PT_ID_by_category.keys()

                rois = self.organs_to_check + [x + '_in_volPerc' for x in self.tissue_comps_to_check]
                for roi in rois: # ex. organ, tissue
                    PT_IDs = []
                    for tPoint in self.tPoints_to_check:  # ex. BL vs PreSurg
                        groups = []
                        temp_tPoint_values = []
                        for key in group_keys:  # ex. High vs Low
                            PT_ID_by_group = PT_ID_by_category[key] # PT_IDs for one group
                            vol_vals_by_group_df = self.volumetric_values_df[self.volumetric_values_df['PT_ID'].isin(PT_ID_by_group)]
                            col_name = tPoint + '_' + roi
                            tPoint_roi_col = vol_vals_by_group_df[col_name].tolist() # Vol vals for one tPoint & one group
                            temp_tPoint_values = temp_tPoint_values + tPoint_roi_col
                            groups = groups + np.repeat(key, len(tPoint_roi_col)).tolist()
                            PT_IDs = PT_IDs + PT_ID_by_category[key].tolist()
                        vol_vals_dict_by_groups[tPoint].append(temp_tPoint_values)
                    vol_vals_dict_by_groups['groups'].append(groups)
                    vol_vals_dict_by_groups['PT_IDs'].append(PT_IDs)
                    vol_vals_dict_by_groups['roi'].append(roi)

                for roi_i in range(len(rois)):  # ex. organ, tissue
                    roi = rois[roi_i]
                    if roi_i <2:
                        y_label = 'mm3'
                    else:
                        y_label = 'Vol%'

                    linepolot_dirPath = os.path.join(self.save_dirPath, "stat_lineplots")
                    if not os.path.exists(linepolot_dirPath):
                        os.makedirs(linepolot_dirPath)
                    pt_by_pt_fName = str(endpoint) + "_" + str(roi) + "_line_plot_pt_by_pt.png"
                    gr_by_gr_fName = str(endpoint) + "_" + str(roi) + "_line_plot_gr_by_gr.png"
                    draw_line_plot_by_group(data_1 = vol_vals_dict_by_groups['BL'][roi_i], data_2 = vol_vals_dict_by_groups['PreSurg'][roi_i],
                                            groups=vol_vals_dict_by_groups['groups'][roi_i], PT_ID_labels = vol_vals_dict_by_groups['PT_IDs'][roi_i],
                                            fig_title = vol_vals_dict_by_groups['roi'][roi_i],y_label = y_label,
                                            save_fPath=os.path.join(linepolot_dirPath, pt_by_pt_fName))
                    draw_line_plot_by_group_gr_by_gr(data_1 = vol_vals_dict_by_groups['BL'][roi_i], data_2 = vol_vals_dict_by_groups['PreSurg'][roi_i],
                                            groups=vol_vals_dict_by_groups['groups'][roi_i], PT_ID_labels = vol_vals_dict_by_groups['PT_IDs'][roi_i],
                                            fig_title = vol_vals_dict_by_groups['roi'][roi_i],y_label = y_label,
                                            save_fPath=os.path.join(linepolot_dirPath, gr_by_gr_fName))











def post_processing_vol_vals_from_df(volumetric_valus_df_fPath:str):
    if volumetric_valus_df_fPath.endswith("csv"):
        volumetric_vals_df = pd.read_csv(volumetric_valus_df_fPath)
    else:
        print("The volumetric_vals_df should be stored as csv file. Not excel workbook. ")


if __name__ == "__main__":

    ### Control variables.

    ### Inherited from module 03.
    tPoints_to_check = ['BL', 'PreSurg']
    organ_vol_calc_option = 1 # 0: Auto / 1: Manually from the metadata.
    organs_to_check = ['liver', 'pancreas']
    tissue_comps_to_check = ['subcutaneous_fat', 'torso_fat', 'skeletal_muscle']
    read_dirPath = "/rsrch5/home/radphys_rsch/ykim23/Pancreatic_Cancer/Converted_Nifti/sorted_pts_with_BL_and_PreSurg_fin/temp"
    save_dirPath = "/rsrch5/home/radphys_rsch/ykim23/Pancreatic_Cancer/Converted_Nifti/sorted_pts_with_BL_and_PreSurg_fin/stat_results"
    #save_fName = "temp_volumetric_vals_df.csv"
    is_save_img_at_center_L3 = True # True: Will save img arrays at center L3.

    ### Originally for module 04.
    load_prepcoessed_df_fName = "temp_volumetric_vals_df_72pts.csv"
    read_scoring_df_fPath = "/home/ykim23/Pancreatic_Cancer/Alliance_Trial/Working_Sheets/1_Notes_Sheets/tmp_verified_72pts.csv"
    PT_ID_colName_in_scoring_df = "Patient Identfier"
    save_fName_post_proc_df = "volumetric_vals_df.csv" # For the processed df.
    save_fName_stat_anal_df = "_statistics.csv" # For the statistics df.
    endpoints_to_check = {"Consensus_plus_alpha_Delta_Score":["High Delta", "Low Delta"],
                          "Consensus_plus_Interface_at_Presurg":["Type I", "Type II"]}
    is_save_boxplot = True
    vol_df = post_processed_volumetric_df()
    vol_df.set_tPoints_organs_tissue_branches(tPoints_to_check=tPoints_to_check, organs_to_check=organs_to_check,
                                              tissue_comps_to_check=tissue_comps_to_check)
    vol_df.set_read_and_save_dirPath(read_dirPath=read_dirPath, save_dirPath=save_dirPath, read_scoring_df_fPath=read_scoring_df_fPath, load_prepcoessed_df_fName=load_prepcoessed_df_fName, save_fName_post_proc_df=save_fName_post_proc_df, save_fName_stat_anal_df=save_fName_stat_anal_df)
    vol_df.run_post_processing()

    vol_df.initialize_for_stat_analysis(read_scoring_df_fPath=read_scoring_df_fPath, PT_ID_colName_in_scoring_df=PT_ID_colName_in_scoring_df, endpoints_to_check=endpoints_to_check, is_save_boxplot=is_save_boxplot)
    vol_df.run_stats()


