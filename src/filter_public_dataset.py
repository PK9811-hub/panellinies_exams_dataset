import pandas as pd

df = pd.read_excel("results/panellinies_dataset.xlsx")

df_public = df[df['year'] != 2026]

df_public.to_excel("results/panellinies_dataset_public.xlsx", index=False)

print(f"The filtered dataset has been created! The questions from 2026 have been removed. Total rows: {len(df_public)}")