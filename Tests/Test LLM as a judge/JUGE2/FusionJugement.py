import pandas as pd

# Charger les fichiers Excel
df_verdict1 = pd.read_excel("verdicts_llm.xlsx")
df_verdict2 = pd.read_excel("verdicts_llm2.xlsx")
df_verdict3 = pd.read_excel("verdicts_llm3.xlsx")

# Concaténer horizontalement
df_concat = pd.concat([df_verdict1, df_verdict2[['Echantillon_exp2', 'Echantillon_verite2', 'verdict_llm2']],
                       df_verdict3[['Echantillon_exp3', 'Echantillon_verite3', 'verdict_llm3']]], axis=1)


# Réorganiser les colonnes selon l'ordre désiré
colonnes_finales = ['Titre_exp', 'Titre_verite','score_titre', 'Echantillon_exp1', 'Echantillon_verite1', "verdict_llm1", 'Echantillon_exp2', 'Echantillon_verite2', 'verdict_llm2', 'Echantillon_exp3', 'Echantillon_verite3', 'verdict_llm3']
df_concat = df_concat[colonnes_finales]

# Sauvegarder dans un nouveau fichier Excel
df_concat.to_excel("Jugement.xlsx", index=False)