import pandas as pd

df = pd.read_excel("results/panellinies_dataset.xlsx")

df_public = df[df['year'] != 2026]

df_public.to_excel("results/panellinies_dataset_public.xlsx", index=False)

print(f"Το φιλτραρισμένο dataset δημιουργήθηκε! Αφαιρέθηκαν οι ερωτήσεις του 2026. Σύνολο γραμμών: {len(df_public)}")