import pandas as pd
import sys
import os
import unicodedata
import subprocess




# Ajoute le dossier parent de RAG2/ (c’est-à-dire RAG/) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UniversityLLMAdapter import UniversityLLMAdapter

'''subprocess.run([sys.executable, "LLM_AAJ.py"])
subprocess.run([sys.executable, "LLM_AAJ2.py"])

subprocess.run([sys.executable, "LLM_AAJ3.py"])

subprocess.run([sys.executable, "FusionJugement.py"])'''

df1 = pd.read_excel("verdicts_llm.xlsx")
df2 = pd.read_excel("verdicts_llm2.xlsx")
df3 = pd.read_excel("verdicts_llm3.xlsx")

def analyser_verdict(df, verdict_col):
    stats = {}
    lignes_total = len(df)
    articles_uniques = df['ID_article'].nunique()

    grouped = df.groupby('ID_article')
    oui_par_article = 0

    lignes_par_article = []

    for article_id, group in grouped:
        nb_lignes = len(group)
        nb_oui = group[verdict_col].str.lower().eq('Oui').sum()
        nb_non = group[verdict_col].str.lower().eq('Non').sum()
        est_oui = nb_oui > 0

        if est_oui:
            oui_par_article += 1

        lignes_par_article.append({
            'ID_article': article_id,
            'Nb_lignes': nb_lignes,
            'Oui': nb_oui,
            'Non': nb_non,
            'Verdict_final': "Oui" if est_oui else "Non"
        })

    pourcentage_oui = (oui_par_article / articles_uniques) * 100

    return {
        "Nombre total de lignes": lignes_total,
        "Nombre d'articles différents": articles_uniques,
        "Détails par article": lignes_par_article,
        "Articles considérés comme Oui": oui_par_article,
        "Pourcentage Oui": round(pourcentage_oui, 2)
    }


resultats1 = analyser_verdict(df1, "verdict_llm1")
resultats2 = analyser_verdict(df2, "verdict_llm2")
resultats3 = analyser_verdict(df3, "verdict_llm3")

df = assigner_ids_articles(df1, colonne_titre="Titre_verite")
print(df[["Titre_verite", "ID_article"]])

df = assigner_ids_articles(df2, colonne_titre="Titre_verite")
print(df[["Titre_verite", "ID_article"]])

df = assigner_ids_articles(df3, colonne_titre="Titre_verite")
print(df[["Titre_verite", "ID_article"]])

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


def formatter_resultats(resultats, titre):
    lignes = [f"📌 {titre}",
              f"- Nombre total de lignes : {resultats['Nombre total de lignes']}",
              f"- Nombre d'articles différents : {resultats['Nombre d\'articles différents']}",
              f"- Nombre d'articles considérés comme Oui : {resultats['Articles considérés comme Oui']} "
              f"({resultats['Pourcentage Oui']}%)",
              "\n[Statistiques par article]"]

    for art in resultats["Détails par article"]:
        lignes.append(
            f"  - Article {art['ID_article']} : "
            f"{art['Nb_lignes']} lignes, {art['Oui']} Oui / {art['Non']} Non → Verdict : {art['Verdict_final']}"
        )

    return "\n".join(lignes)

# Export
with open("Stat_Echantillon.txt", "w", encoding="utf-8") as f:
    f.write(formatter_resultats(resultats1, "Bilan LLM1"))

with open("Stat_Protocole.txt", "w", encoding="utf-8") as f:
    f.write(formatter_resultats(resultats2, "Bilan LLM2"))

with open("Stat_Conclusion.txt", "w", encoding="utf-8") as f:
    f.write(formatter_resultats(resultats3, "Bilan LLM3"))


