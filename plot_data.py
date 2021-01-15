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
    ycolSuc = 'Succes Percentage'

    fig1, ax1 = plt.subplots()
    sns.lineplot(ax = ax1, data=model_res, x=xcolFPR, y=ycolTPR, marker='o', color='red')
    plt.xticks(np.arange(0.0, 1.2, 0.2))
    ax1.set(ylabel='TP rate', xlabel='FP rate')

    area = auc(model_res["FPR"].to_numpy(), model_res["TPR"].to_numpy())
    plt.annotate('AUC: {}'.format(area), xy=(0.3,0.5))
    
    fig2, ax2 = plt.subplots() #1,2,figsize=(16,6)
    sns.lineplot(ax = ax2, data=model_res, x= xcolLim, y= ycolAcc,marker = 'o')
    sns.lineplot(ax = ax2, data=model_res, x= xcolLim, y= ycolSuc, color ='red',marker='o')

    ax2.set(ylabel='Percent', xlabel='Score limit')
    ax2.legend([ycolAcc, 'Success Percentage'])



    print(model_res)


    # fig2, ax2 = plt.subplots(1,3,figsize=(16,4))
    # sns.lineplot(ax = ax2[0],data=model_res, y= 'Loss')
    # sns.lineplot(ax = ax2[1],data=model_res, y= 'Succes Percentage',color ='red')
    # sns.lineplot(ax = ax2[2],data=model_res, y= 'True positives',color = 'green')
    # ax2.set()
    # fig2.suptitle('boooooi')

    plt.show()

def main():
    plotResults()


if __name__ == "__main__":
    main()