import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm  # Pour une barre de progression

# Chargement des fichiers
fichier_resultats = "fichier_fusionne.xlsx"
fichier_verite = "FinalRawData.xlsx"

# Chargement des colonnes pertinentes
df_resultats = pd.read_excel(fichier_resultats, usecols=["Titre", "Extrait pertinent"])
df_verite = pd.read_excel(fichier_verite, usecols=["Title", "Excerpt"])

# Normalisation
df_resultats["Titre_norm"] = df_resultats["Titre"].str.lower().str.strip()
df_verite["Title_norm"] = df_verite["Title"].str.lower().str.strip()

titres_verite = df_verite["Title_norm"].tolist()

correspondances = []

for i, titre_res in enumerate(df_resultats["Titre_norm"]):
    if pd.isna(titre_res) or not titre_res.strip():
        correspondances.append({"index_resultat": i, "index_verite": None, "score": 0})
        continue

    result = process.extractOne(titre_res, titres_verite, scorer=fuzz.token_sort_ratio)

    if result:
        match, score, index = result
        correspondances.append({
            "index_resultat": i,
            "index_verite": index,
            "score": score
        })
    else:
        correspondances.append({
            "index_resultat": i,
            "index_verite": None,
            "score": 0
        })

# Construction des DataFrames appariés
df_resultats_matched = df_resultats.copy()
df_verite_matched = pd.DataFrame(
    [df_verite.iloc[c["index_verite"]] if c["index_verite"] is not None else pd.Series(dtype='object') for c in correspondances]
)
scores_titre = [c["score"] for c in correspondances]

# Réinitialiser les index
df_resultats_matched.reset_index(drop=True, inplace=True)
df_verite_matched.reset_index(drop=True, inplace=True)

print(f"✅ Appariement terminé : {len(df_resultats_matched)} lignes appariées")

import sys
import os

# Ajoute le dossier parent de RAG2/ (c’est-à-dire RAG/) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Initialisation du LLM ===
from UniversityLLMAdapter import UniversityLLMAdapter  # adapte ce chemin selon ton projet

llm = UniversityLLMAdapter(
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImRjZjZkYWE2LTRjYzMtNDYyOS05MjJiLTkyYzM1NGQzODYwMCJ9.04oTM2nVW8iQTCr8qrs4MknI13dGgqBp85Wq7t2jAeQ",
    api_url="http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions",
    max_tokens=1024,
    temperature=0.1
)

# === Comparaison via LLM ===
verdicts = []

for idx in tqdm(range(len(df_resultats_matched))):
    ligne_exp = df_resultats_matched.loc[idx]
    ligne_verite = df_verite_matched.loc[idx]

    prompt = f"""
Tu es un assistant scientifique. On te donne deux types d'extrait d'un article de biologie :
- Une ligne expérimentale issue d'un traitement automatique qui a pour but de trouver les protocoles de prélévement ADN 
- Une ligne de référence qui a été extrait à la main par des spècialiste 

Ta tâche est de juger si la ligne donner par le traitement automatique est pertinante sachant que l'on cherche des protocoles de prélévement ADN et si il y a corrzspondance avec l'extrait prélever à la main.
Si l'information expérimentale est plus précise mais reste cohérente avec la vérité terrain, cela compte comme une correspondance correcte donc dis OUI.

Réponds seulement par "Oui" ou "Non", puis justifie brièvement en une ou deux phrases.

--- LIGNE EXPÉRIMENTALE ---
Titre : {ligne_exp['Titre']}
Extrait : {ligne_exp['Extrait pertinent']}


--- LIGNE DE VÉRITÉ ---
Titre : {ligne_verite['Title']}
Extrait : {ligne_verite['Excerpt']}

Ces deux lignes parlent-elles du même échantillon biologique ?
"""

    verdict = llm.generate_answer(prompt)
    verdicts.append(verdict.strip())

# Fusion des deux DataFrames + verdict
df_final = pd.DataFrame({
    "Titre_exp": df_resultats_matched["Titre"],
    "Echantillon_exp2": df_resultats_matched["Extrait pertinent"],
    "Titre_verite": df_verite_matched["Title"],
    "Echantillon_verite2": df_verite_matched["Excerpt"],
    "score_titre": scores_titre,
    "verdict_llm2": verdicts
})

# Sauvegarde du fichier fusionné
df_final.to_excel("verdicts_llm2.xlsx", index=False)

print("✅ Comparaison LLM terminée.")
