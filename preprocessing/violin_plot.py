import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
def rename_dataframe(departure,arrival):

    path_HFE = "Database/HFE/"+departure+"/"

    # for root, dirs, files in os.walk("/Users/aarc8/Documents/github/DataProcess/"+path_HFE):
    #     for file in files:
    #         if file.endswith("data_HEF_"+departure+arrival+".csv"):
    #             print(os.path.join(root, file))
    
    df_new= pd.read_csv("Database/HFE/"+departure+"/"+"data_HEF_"+departure+arrival+".csv")
    # print(df_new.head())
    # df_new = dataframe.HFE
    df_new = df_new.rename(columns = {"HFE":arrival})

    df_new = df_new.iloc[: , 1:]

    # print(df_new.head())

    df_new = df_new[arrival]

    return(df_new)

# departures = ['FRA','LHR','CDG','AMS','MAD','BCN','FCO','DUB','VIE','ZRH','ARN','DME','HEL','IST','KBP']

departures = ['LHR']
arrivals = ['FRA','CDG','AMS','MAD','BCN','FCO','DUB','VIE','ZRH','ARN','DME','HEL','IST','KBP']



for i in range(len(departures)):
    for j in range(0,len(arrivals)-1):
        # if (i != j):
        # if (i != j) and (i > 6) and (i < 8) and (j>11) and (j<13):
        # if (i != j) and (i > 6) and (i < 8):
        if j == 0:
            df_1 = rename_dataframe(departures[i], arrivals[j])
            df_2 = rename_dataframe(departures[i], arrivals[j+1])
            df_final = pd.concat([df_1,df_2],axis=1)
        else:
            df_2 = rename_dataframe(departures[i], arrivals[j+1])
            df_final = pd.concat([df_final,df_2],axis=1)

                

            print(df_final.head())

            

# dataframe = pd.read_csv("Database\HFE\AMS\data_HEF_AMSIST.csv", error_bad_lines=False, encoding="ISO-8859-1")
# dataframe2 = pd.read_csv("Database\HFE\AMS\data_HEF_AMSIST.csv", error_bad_lines=False, encoding="ISO-8859-1")

df_final = df_final[df_final.notna()]

df_mask=df_final < 100
df_final = df_final[df_mask]
# df_final= df_final.loc[((df_final>50)).any(1)]
print(df_final.info())
print(df_final.describe())
# population = dataframe.HFE

sns.set_theme(style="whitegrid")

# tips = sns.load_dataset("tips")

# ax = sns.violinplot(x=population)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=df_final, cut=0, scale='width', bw=0.25)

_ = plt.xticks(rotation=45, ha='right')
sns.despine(left=True)

ax.set_title('Departing from '+departures[0])
ax.set_ylabel('HEF [%]')
plt.yticks(np.arange(0, 110, 10))

plt.show()