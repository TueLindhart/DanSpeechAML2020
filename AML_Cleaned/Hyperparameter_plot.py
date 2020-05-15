
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
path = os.path.join(os.getcwd(),"Scores","allScores-hyperparam.csv")

for_noisy_df = pd.read_csv(path,sep=";")
for_noisy_df = for_noisy_df.fillna("Room_reverb_Baseline_Noisy")
for_means_noisy = for_noisy_df.groupby(["score_ID","Augmentation_list"]).mean()[["WER"]]
df_tuning = for_means_noisy.sort_values(by="WER",ascending=True).copy()
df_tuning = df_tuning[df_tuning["WER"]>0.3]
df_tuning = df_tuning.reset_index()
df_temp = df_tuning[df_tuning["Augmentation_list"]=="['spec_augment']"]
df_temp.reset_index()

df_room_base = df_tuning[df_tuning["score_ID"]=="Room_reverb_Baseline_Noisy"]
df_room_base=df_room_base[df_room_base["Augmentation_list"]=="['room_reverb']"]
room_base = df_room_base["WER"].values*100    
    
df_room = df_tuning[df_tuning["Augmentation_list"]=="['room_reverb']"]
df_room.head()

room_val = np.array(df_room["WER"])*100
room_mean = room_val.mean()
room_var = room_val.var()
room_min = room_val.min()
room_max = room_val.max()

df_specAug = df_tuning[df_tuning["Augmentation_list"]=="['spec_augment']"]
df_specAug.head()

specAug_val = np.array(df_specAug["WER"])*100
specAug_mean = specAug_val.mean()
specAug_var = specAug_val.var()
specAug_min = specAug_val.min()
specAug_max = specAug_val.max()

df_noisy_one = for_noisy_df[for_noisy_df["score_ID"]=="Noisy_hyperparameter_tuning__0.13_0.36_3.19_0_5_7.49_0_5_7.08_0_0.5_0.5_0.5_0.5"]
df_noisy_one = df_noisy_one[df_noisy_one["Augmentation_list"]=="['room_reverb']"]

df_noisy_base = for_noisy_df[for_noisy_df["score_ID"]=="Room_reverb_Baseline_Noisy"]
df_noisy_base = df_noisy_base[df_noisy_base["Augmentation_list"]=="['room_reverb']"]

df_noisy_zero = for_noisy_df[for_noisy_df["score_ID"]=="Noisy_specAug_standard_parameters"]
df_noisy_zero = df_noisy_zero[df_noisy_zero["Augmentation_list"]=="['room_reverb']"]

df_noisy_worst = for_noisy_df[for_noisy_df["score_ID"]=="Noisy_hyperparameter_tuning__0.37_0.58_4.31_0_5_3.89_0_5_3.89_0_0.5_0.5_0.5_0.5"]
df_noisy_worst = df_noisy_worst[df_noisy_worst["Augmentation_list"]=="['room_reverb']"]

val_dict = {"Worst": df_noisy_worst["WER"]*100,
            "Baseline v2": df_noisy_zero["WER"]*100,
            "Baseline": df_noisy_base["WER"]*100,
            "Best": df_noisy_one["WER"]*100}

mean_dict = {"Worst": df_noisy_worst["WER"].mean()*100,
            "Baseline v2": df_noisy_zero["WER"].mean()*100,
            "Baseline": df_noisy_base["WER"].mean()*100, 
            "Best": df_noisy_one["WER"].mean()*100}


fig1, ax1 = plt.subplots()
ax1.set_title('Hyparparameter tuning for room reverberation' )

ax1.plot([x for x in mean_dict.values()],[1,2,3,4],marker='o')
ax1.boxplot(val_dict.values(),vert=False)

#ax1.boxplot(room_val,vert=False)
ax1.set_xlabel("Word error rate (%) (WER)")
ax1.set_yticklabels(val_dict.keys())
ax1.set_ylabel("")
plt.show()


names = [x for x in mean_dict.keys()]
val = [x for x in mean_dict.values()]
print("Mean WER of different models:")
for i in range(len(names)):
    print("%20s\t%.2f"%(names[len(names)-i-1],val[len(names)-i-1]))
