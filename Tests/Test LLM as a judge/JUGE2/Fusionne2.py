import pandas as pd

# Charger les fichiers Excel
df_protocoles = pd.read_excel("protocoles_117_Fichiers.xlsx")
df_articles = pd.read_excel("articles_grobid.xlsx")

# Renommer la colonne 'Filename' pour qu'elle corresponde à 'Fichier'
df_protocoles = df_protocoles.rename(columns={'Filename': 'Fichier'})

# Fusionner sur la colonne 'Fichier'
df_merged = pd.merge(df_articles, df_protocoles, on='Fichier', how='inner')

# Réorganiser les colonnes selon l'ordre désiré
colonnes_finales = ['Fichier', 'Titre', 'Protocole', 'Extrait pertinent', "Évaluation d'invasivité", 'Péchés identifiés', 'Nouveaux péchés']
df_merged = df_merged[colonnes_finales]

# Sauvegarder dans un nouveau fichier Excel
df_merged.to_excel("fichier_fusionne.xlsx", index=False)
