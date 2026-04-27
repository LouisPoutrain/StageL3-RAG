import pandas as pd
import sys
import os
import unicodedata
import subprocess




# Ajoute le dossier parent de RAG2/ (c’est-à-dire RAG/) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UniversityLLMAdapter import UniversityLLMAdapter

subprocess.run([sys.executable, "LLM_AAJ.py"])
subprocess.run([sys.executable, "LLM_AAJ2.py"])

subprocess.run([sys.executable, "LLM_AAJ3.py"])

subprocess.run([sys.executable, "FusionJugement.py"])


# === Chargement du fichier contenant les 3 verdicts ===
df = pd.read_excel("Jugement.xlsx")

# Réduction éventuelle si besoin (tu peux supprimer cette ligne si tu veux tout analyser)
df_sample = df.head(200)

def normaliser_texte(texte):
    if pd.isna(texte):
        return ""
    # Minuscule + suppression des accents + strip
    texte = texte.lower().strip()
    texte = unicodedata.normalize('NFD', texte)
    texte = ''.join(c for c in texte if unicodedata.category(c) != 'Mn')
    return texte

def assigner_ids_articles(df, colonne_titre="Titre_exp", colonne_id="ID_article"):
    """
    Ajoute une colonne 'ID_article' à un DataFrame en regroupant les titres similaires.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les titres d'article.
        colonne_titre (str): Le nom de la colonne contenant les titres.
        colonne_id (str): Le nom de la colonne à ajouter pour les identifiants d'article.
    
    Returns:
        pd.DataFrame: Le DataFrame enrichi d'une colonne 'ID_article'.
    """
    titres_norm = df[colonne_titre].apply(normaliser_texte)
    df[colonne_id] = titres_norm.factorize()[0]
    return df

df = assigner_ids_articles(df_sample, colonne_titre="Titre_verite")
print(df[["Titre_verite", "ID_article"]])

# Préparation des données lisibles pour le LLM
table_lisible1 = ""
table_lisible2 = ""
table_lisible3 = ""

for i, row in df_sample.iterrows():
    table_lisible1 += f"""
⎯⎯⎯ Entrée {i+1} ⎯⎯⎯
ID : {row['ID_article']}
Titre VERITE : {row['Titre_verite']}
Verdict : {row['verdict_llm1']}"""
    table_lisible2 += f"""
⎯⎯⎯ Entrée {i+1} ⎯⎯⎯
ID : {row['ID_article']}
Titre VERITE : {row['Titre_verite']}
Verdict : {row['verdict_llm2']}"""
    table_lisible3 += f"""
⎯⎯⎯ Entrée {i+1} ⎯⎯⎯
ID : {row['ID_article']}
Titre VERITE : {row['Titre_verite']}
Verdict : {row['verdict_llm3']}"""
    

# === Prompt pour le second LLM ===
prompt_template = """
Tu es un assistant scientifique spécialisé en validation de résultats expérimentaux.

On te fournit un tableau avec le verdict d’un premier LLM .

Ta mission est d’analyser les résultats de ce LLM de manière synthétique et rigoureuse. Plusieurs lignes peuvent concerner un même article — regroupe ton analyse en tenant compte de cela.

Attention les titres ne doivent pas ètre pris en compte. Tu dois te baser UNIQUEMENT sur les verdict.

Compte bien le nombre d'article différent en te basant sur les ID. (Si l'ID max est 6 alors il y a 7 articles)

Réponds strictement avec la structure suivantes :

Description briève de ce qui es comparée.

[Statistiques]
- Nombre total de lignes
- Nombre d'articles différent 
- Nombre et pourcentage de "Oui"/"Non" par article : 
    - Article 1 : <Nom> :
            Nombre de ligne, Nombre de "Oui"/Nombre de "Non" ,Si il y a un "Oui" alors considerer cette article comme "Oui".
    - Article 2 : <Nom> :
            Nombre de ligne, Nombre de "Oui"/Nombre de "Non" , Si il y a un "Oui" alors considerer cette article comme "Oui". 
    ...
-Nombre d'article considérer comme "Oui" et pourcentage 



Voici les données à analyser :

{table}
"""

# === Envoi au LLM ===
llm = UniversityLLMAdapter(
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImRjZjZkYWE2LTRjYzMtNDYyOS05MjJiLTkyYzM1NGQzODYwMCJ9.04oTM2nVW8iQTCr8qrs4MknI13dGgqBp85Wq7t2jAeQ",
    api_url="http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions",
    max_tokens=4096,
    temperature=0
)

prompt1 = prompt_template.format(table=table_lisible1)
prompt2 = prompt_template.format(table=table_lisible2)
prompt3 = prompt_template.format(table=table_lisible3)


reponse_bilan1 = llm.generate_answer(prompt1)
reponse_bilan2 = llm.generate_answer(prompt2)
reponse_bilan3 = llm.generate_answer(prompt3)


# === Affichage et sauvegarde ===
print("📊 BILAN DU  LLM :\n")
print(reponse_bilan1)
print(reponse_bilan2)
print(reponse_bilan3)


with open("Stat_Echantillon.txt", "w", encoding="utf-8") as f:
    f.write(reponse_bilan1)

with open("Stat_Protocole.txt", "w", encoding="utf-8") as f:
    f.write(reponse_bilan2)

with open("Stat_Conclusion.txt", "w", encoding="utf-8") as f:
    f.write(reponse_bilan3)
