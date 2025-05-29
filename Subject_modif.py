import pandas as pd
df = pd.read_excel('Subject_details.xlsx')

def label(score):
    if score >= 10:
        return 1
    else:
        return 0
df['label'] = df['Score'].apply(label)
print(df.head())
data = df[['Data Id', 'label']].copy()
print(data)

data.to_csv("labels_processed.csv", index=False)
