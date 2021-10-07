import pandas as pd
import seaborn as sns

for idx in range(5):
    filePath = f'./run-{idx}-tag-Acc.csv'
    if idx:
        df_index = pd.read_csv(filePath)
        df_index['Time'] = idx
        df = df.append(df_index)
    else:
        df_index = pd.read_csv(filePath)
        df_index['Time'] = idx
        df = df_index
    
    df.drop(['Wall time'], axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)

NameMap = {'Value': 'Acc', 'Step': 'Epoch'}

df = df.rename(columns=NameMap)

sns.set_style("white")

palette = sns.color_palette("bright", 5)
fig = sns.lineplot(data=df, x="Epoch", y="Acc", hue='Time', palette=palette)

line_fig = fig.get_figure()
line_fig.savefig('../imgs/lineplot.png', dpi=400)
