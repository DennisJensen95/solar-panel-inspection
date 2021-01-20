import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import os
from sklearn.metrics import auc

def setup_arg_parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('--debug', help='debug flag help')
    parser.add_argument('--filepath', help='Data file path')
    args = parser.parse_args()
    
    return args

def plotResults():
    args = setup_arg_parsing()

    if args.filepath == None:
        print(f'Please specify a model')
        return
    else:
        path = args.filepath
        if not os.path.exists(path):
            print(f'Folder does not exist')
            return
        
        try:
            filename = os.path.basename(os.path.normpath(path)) # extract folder name
            model_res = pd.read_csv(path)
        except:
            print(f'Failed to load model')
            return

    image_folder = "Image_folder"
    name = os.path.basename(os.path.normpath(filename))
    path_im = create_im_folder(name,image_folder)

    sns.set_theme()

    l = np.arange(len(model_res))
    model_res["epoch"] = list(l)

    Tpos = model_res["True positives"].to_numpy()
    Fpos = model_res["False positives"].to_numpy()
    Fneg = model_res["False negatives"].to_numpy()
    Tneg = model_res["True negatives"].to_numpy()

    # TPR = TP/(TP+FN)
    Tpr = np.divide(Tpos, np.add(Tpos, Fneg))
    Fpr = np.divide(Fpos, np.add(Fpos, Tneg))

    model_res["TPR"] = Tpr
    model_res["FPR"] = Fpr

    # model_res["Limits"] = np.linspace(0.0,1.0,num=21)

    xcolLim = 'Limits'
    ycolTPR = 'TPR'
    xcolFPR = 'FPR'
    ycolAcc = 'Accuracy'
    ycolSucF = 'Success (fault)'
    ycolSucNF = 'Success (no fault)'

    fig1, ax1 = plt.subplots()
    sns.lineplot(ax = ax1, data=model_res, x=xcolFPR, y=ycolTPR, marker='o', color='red')
    plt.xticks(np.arange(0.0, 1.2, 0.2))
    ax1.set(ylabel='TP rate', xlabel='FP rate')

    area = auc(model_res["FPR"].to_numpy(), model_res["TPR"].to_numpy())
    plt.annotate('AUC: {}'.format(area), xy=(0.3,0.5))

    save_fig(fig1, path_im, "roc_plot")

    
    fig2, ax2 = plt.subplots() #1,2,figsize=(16,6)
    sns.lineplot(ax = ax2, data=model_res, x= xcolLim, y= ycolAcc,marker = 'o')
    sns.lineplot(ax = ax2, data=model_res, x= xcolLim, y= ycolSucNF, color ='green',marker='o')
    sns.lineplot(ax = ax2, data=model_res, x= xcolLim, y= ycolSucF, color ='red',marker='o')

    ax2.set(ylabel='Percent', xlabel='Score limit')
    ax2.legend([ycolAcc, ycolSucNF, ycolSucF])

    save_fig(fig2, path_im, "accuracy_plot")

    print(model_res)

    plt.show()

def save_fig(fig, path, des):
    filename = path+"/"+des+".png"
    fig.savefig(filename)

def create_im_folder(name,image_folder):
    create_image_folder(image_folder)
    folder_name = name
    path = image_folder + "/" + folder_name
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_image_folder(image_folder):
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

def main():
    plotResults()


if __name__ == "__main__":
    main()