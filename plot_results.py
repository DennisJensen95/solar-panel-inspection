import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme()
path = "./Results-folder/"
folder = "solar_model_mask_fault-classification_20210114-093235/"
file = "solar_model_data_mask_fault-classification_20210114-093235"

yeet = "./Results-folder/solar_model_mask_fault-classification_20210113-220116/solar_model_data_mask_fault-classification_20210113-220116"
model_res = pd.read_csv(path+folder+file)
model_res2 = pd.read_csv(yeet)

batch1 = np.ones(len(model_res), dtype=int)
batch2 = np.ones(len(model_res2), dtype=int)*2
batch1 = np.append(batch1,batch2)

model_res = model_res.append(model_res2)
l = np.arange(len(model_res))
model_res["epoch"] = list(l)
model_res["batch"] = list(batch1)

print(model_res.head)
sns.lineplot(data=model_res, x= 'epoch', y= 'Loss', hue='batch')
plt.show()

# fig, ax = plt.subplots(1,3)
# sns.lineplot(ax = ax[0],data=model_res, x= 'epoch', y= 'Loss', linewidth=2)
# #sns.lineplot(ax = ax[1],data=model_res, x= 'epoch', y= 'Accuracy')
# sns.lineplot(ax = ax[1],data=model_res, x= 'epoch', y= 'Succes Percentage',color ='red')
# sns.lineplot(ax = ax[2],data=model_res, x= 'epoch', y= 'True positives',color = 'green')
# sns.lineplot(ax = ax[1],data=model_res, x= 'epoch', y= 'Accuracy',color = 'blue')

#plt.show()