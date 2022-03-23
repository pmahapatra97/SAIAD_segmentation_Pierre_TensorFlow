import re 
import pandas as pd
import matplotlib.pyplot as plt
import  matplotlib.cm as cm
import os
import numpy as np
import constant
import argparse


def get_results(experiment, CRF = False,
                elements = ["Patient : ","DICE_mean : ","IU_mean : "],
                names = ["Patient","Dice","IoU"]):
    patientList = np.sort([f for f in os.listdir("./results_"+experiment) if re.match("SAIAD*",f)])
    results_list = []
    df_ = pd.DataFrame(columns=names)
    
    for patient in patientList:
        filepath = "results_"+experiment+"/"+patient+"/logs/resultDiceIU.txt"
        if CRF:
            filepath = "results_"+experiment+"/"+patient+"/logs/resultDiceIU_CRF.txt"
        file = open(filepath, "r")
        results = {"Patient":patient}
        for line in file:
            for element, name in zip(elements,names):
                if re.search(element, line):
                    l = line.replace(element,"")
                    l = l.replace("\n","")
                    results[name]=l
                    break
        results_list.append(results)
    # df_res = pd.DataFrame(results_list).set_index("Patient").apply(pd.to_numeric)
    # df_mean = df_res.mean()
    # df_min = df_res.min()
    # df_max = df_res.max()
    # df_
    # df_res.loc["Mean"]= 
    # df_res.loc["Mean"]= df_res.mean()
    df_res.loc["Mean"]= df_res.mean()
    return df_res

def read_results(experiment):
    df_res = pd.read_csv(constant.CSV_FILE.format(experiment),index_col="Patient")
    return df_res

# def show_results(experiments_df, experiments_names, set_text= False):

#     # bar values
#     bar_values = [df.Dice*100 for df in experiments_df]
#     labels = ["Dice "+name for name in experiments_names]

#     n_exp = len(bar_values) 
#     n_samples = len(bar_values[0])
#     barWidth = 0.8/n_exp
#     # x-axis values
#     r_bar = [np.arange(n_samples)+barWidth*j - barWidth*(n_exp-1)/2 for j in range(n_exp)]

#     # cmap = cm.get_cmap('brg')
#     # colors = cmap(np.linspace(0,1,n_exp))
#     # np.random.shuffle(colors)

#     fig, ax = plt.subplots(figsize=(15, 7))

#     def autolabel(rects):
#         for idx,rect in enumerate(bar_plot):
#             height = rect.get_height()
#             ax.text(rect.get_x() + rect.get_width()/2., max(height-3,10),
#                     "{:.2f}".format(height), fontweight = 'bold',
#                     ha='center', va='top', rotation=90,
#                      fontsize = 8,
#                      color = "black",
#                     bbox=dict(facecolor='white', edgecolor="white", pad=3, alpha=0.5))

#     # Make the plot
#     for j in range(n_exp):
#         bar_plot = ax.bar(r_bar[j], bar_values[j], width=barWidth, label=labels[j])
#         if set_text:
#             autolabel(bar_plot)
        
#     # Add xticks on the middle of the group bars
#     plt.xlabel('Patients', fontweight='bold')
#     plt.xticks(range(n_samples), bar_values[0].keys(), rotation = 45, ha='right')
#     plt.xlim([-1,n_samples])
#     plt.ylim([0,100])
#     # Create legend & Show graphic
#     plt.legend(loc ='lower center',framealpha=1)
#     ax.set_axisbelow(True)
#     ax.grid(color='gray',axis="y")
#     ax.grid(color='grey', linestyle='dashed',linewidth=0.5,axis="y",which="minor")
#     ax.minorticks_on()
#     return fig 

def show_results(experiments_df, experiments_names, legend_loc="lower center", set_text= False, paired=False):
    common_patients = list(set([i for df in experiments_df for i in df.index if i in constant.BASE_CASE_PATIENT_NAMES]))
    common_patients.sort()
    
    # bar values
    bar_values = pd.DataFrame([df.loc[[i for i in df.index if i in common_patients]].Dice*100 for df in experiments_df],experiments_names)
    labels = ["Dice "+name for name in experiments_names]
    col_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if paired:
        col_list = cm.get_cmap("tab20")(range(20))
    n_exp = len(bar_values) 
    n_samples = len(common_patients)+1
    barWidth = 0.8/n_exp
    # x-axis values
    r_bar = [np.arange(n_samples)+barWidth*j - barWidth*(n_exp-1)/2 for j in range(n_exp)]

    # cmap = cm.get_cmap('brg')
    # colors = cmap(np.linspace(0,1,n_exp))
    # np.random.shuffle(colors)

    fig, ax = plt.subplots(figsize=(15, 7))

    def autolabel(rects):
        for idx,rect in enumerate(bar_plot):
            height = rect.get_height()
            if height is not None:
                ax.text(rect.get_x() + rect.get_width()/2., max(height-3,10),
                    "{:.2f}".format(height), fontweight = 'bold',
                    ha='center', va='top', rotation=90,
                     fontsize = 8,
                     color = "black",
                    bbox=dict(facecolor='white', edgecolor="white", pad=3, alpha=0.5))

    # Make the plot
    for j, exp in enumerate(experiments_names):
        bar_plot = ax.bar(r_bar[j][:-1][bar_values.loc[exp].notna()], bar_values.loc[exp][bar_values.loc[exp].notna()], width=barWidth, label=labels[j], color = col_list[j])
        #bar_plot = ax.bar(r_bar[j][-1],bar_values[j],width=barWidth, label=labels[j])
        box_plot = ax.boxplot(bar_values.loc[exp].dropna(), 
                    positions = [r_bar[j][-1]], widths=[barWidth],
                    showmeans=True,
                    patch_artist=True,
                    boxprops={'color': col_list[j],'facecolor': col_list[j]},
                    medianprops={'color': "black"},
                    meanprops={'marker':'o',"markerfacecolor":"k",'mec': "k"})
        if set_text:
            autolabel(bar_plot)
        
    # Add xticks on the middle of the group bars
    plt.xlabel('Patients', fontweight='bold')
    plt.xticks(range(n_samples), list(common_patients)+["Summary"], rotation = 45, ha='right')
    plt.xlim([-1,n_samples])
    plt.ylim([0,100])
    # Create legend & Show graphic
    #plt.legend(loc ='lower center',framealpha=1, fontsize = "x-large")
    plt.legend(loc =legend_loc,framealpha=1, fontsize = "x-large")
    ax.set_axisbelow(True)
    ax.grid(color='gray',axis="y")
    ax.grid(color='grey', linestyle='dashed',linewidth=0.5,axis="y",which="minor")
    ax.minorticks_on()
    plt.tick_params(labelright=True)
    return fig, ax

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract experiment results")
    parser.add_argument("experiments", nargs='+')
    args = parser.parse_args()    
    experiments = args.experiments

    for experiment in experiments:
        print("Get result: "+experiment)
        df_res = get_results(experiment)
        df_res.to_csv(constant.CSV_FILE.format(experiment))

