import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm
from ecologits import EcoLogits




# Chargement des fichiers
fichier_resultats = "fichier_fusionne.xlsx"
fichier_verite = "FinalRawData.xlsx"

# Chargement des colonnes pertinentes
df_resultats = pd.read_excel(fichier_resultats, usecols=["Fichier_Article", "echantillon"])
df_verite = pd.read_excel(fichier_verite, usecols=["Title", "What they called non-invaisve SAMPLES"])

# Normalisation
df_resultats["Titre_norm"] = df_resultats["Fichier_Article"].str.lower().str.strip()
df_verite["Title_norm"] = df_verite["Title"].str.lower().str.strip()

titres_verite = df_verite["Title_norm"].tolist()

correspondances = []

for i, titre_res in enumerate(df_resultats["Titre_norm"]):
    if pd.isna(titre_res) or not titre_res.strip():
        continue  # Ignore les lignes vides

    result = process.extractOne(titre_res, titres_verite, scorer=fuzz.token_sort_ratio)

    if result and result[1] >= 50:
        match, score, index = result
        correspondances.append({
            "index_resultat": i,
            "index_verite": index,
            "score": score
        })

# Construction des DataFrames appariés
df_resultats_matched = pd.DataFrame([df_resultats.iloc[c["index_resultat"]] for c in correspondances])
df_verite_matched = pd.DataFrame([df_verite.iloc[c["index_verite"]] for c in correspondances])
scores_titre = [c["score"] for c in correspondances]

# Réinitialiser les index
df_resultats_matched.reset_index(drop=True, inplace=True)
df_verite_matched.reset_index(drop=True, inplace=True)

print(f"✅ Appariement terminé : {len(df_resultats_matched)} lignes avec score ≥ 75")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

verdicts = []
impacts_energy = []
impacts_co2 = []


# === Initialisation du LLM ===
from UniversityLLMAdapter import UniversityLLMAdapter

llm = UniversityLLMAdapter(
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImZlYjZlNTMwLTFiMmYtNDU3MC04NGIyLTYwZGNkYmZiMDE1MyJ9.skdoGmv6mdM5C2bURzy6WpOfWTlpFPSEGlvZ8U_4F1Q",
    api_url="http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions",
    max_tokens=10000,
    temperature=0.1
)

# === Comparaison via LLM ===
verdicts = []

for idx in tqdm(range(len(df_resultats_matched))):
    ligne_exp = df_resultats_matched.loc[idx]
    ligne_verite = df_verite_matched.loc[idx]

    prompt = f"""
Tu es un assistant scientifique. On te donne deux types d'échantillon :
- Une ligne expérimentale issue d'un traitement automatique qui donne une description de ce qui a été prélevé 
- Une ligne de référence considérée comme la vérité terrain en un seul mot 

Ta tâche est de juger si les deux lignes décrivent *essentiellement* le même échantillon. 
Si l'information expérimentale est plus précise mais reste cohérente avec la vérité terrain, cela compte comme une correspondance correcte donc dis OUI.
Si une ligne mentionne 2 éléments et que l'autre mentionne 1 des 2 alors tu peux répondre "Oui" (ex : A+B et B alors "Oui").

Réponds seulement par "Oui" ou "Non".

--- LIGNE EXPÉRIMENTALE ---
Echantillon : {ligne_exp['echantillon']}


--- LIGNE DE VÉRITÉ ---
Echantillon : {ligne_verite['What they called non-invaisve SAMPLES']}

Ces deux lignes parlent-elles du même échantillon biologique ?
"""

    verdict, impact = llm.generate_answer_with_impact(prompt)
    verdicts.append(verdict)
    impacts_energy.append(impact["energy_kwh"])
    impacts_co2.append(impact["co2_g"])           # en kg CO2eq


# Fusion des deux DataFrames + verdict
df_final = pd.DataFrame({
    "Titre_exp": df_resultats_matched["Fichier_Article"],
    "Echantillon_exp1": df_resultats_matched["echantillon"],
    "Titre_verite": df_verite_matched["Title"],
    "Echantillon_verite1": df_verite_matched["What they called non-invaisve SAMPLES"],
    "score_titre": scores_titre,
    "verdict_llm1": verdicts,
    "energy_kwh": impacts_energy,
    "co2_g": impacts_co2
})


# Sauvegarde
df_final.to_excel("verdicts_llm.xlsx", index=False)

# Calcul des totaux
total_energy = sum(impacts_energy)
total_co2 = sum(impacts_co2)

# === Enregistrement de l'impact dans un fichier texte ===
def enregistrer_impact(nom_script, energie_kwh, co2_g, chemin="impact_llm.txt"):
    ligne = (
        f"📁 Script exécuté : {nom_script}\n"
        f"  - Énergie totale consommée : {energie_kwh:.4f} kWh\n"
        f"  - CO₂ total émis : {co2_g:.2f} g\n"
        "--------------------------------------\n"
    )
    with open(chemin, "a", encoding="utf-8") as f:
        f.write(ligne)

# Récupération du nom du fichier script
nom_du_script = os.path.basename(__file__) if '__file__' in globals() else "Script interactif (ex: Jupyter)"

# Appel à la fonction d'enregistrement
enregistrer_impact(nom_du_script, total_energy, total_co2)

print("✅ Comparaison LLM terminée (seulement les scores ≥ 75).")
