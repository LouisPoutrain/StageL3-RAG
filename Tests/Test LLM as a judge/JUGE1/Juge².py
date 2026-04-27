import pandas as pd
from UniversityLLMAdapter import UniversityLLMAdapter

# === Chargement du fichier contenant les 3 verdicts ===
df = pd.read_excel("Jugement.xlsx")

# Réduction éventuelle si besoin (tu peux supprimer cette ligne si tu veux tout analyser)
df_sample = df.head(200)

# Préparation des données lisibles pour le LLM
table_lisible = ""
for i, row in df_sample.iterrows():
    table_lisible += f"""
⎯⎯⎯ Entrée {i+1} ⎯⎯⎯
Titre EXP : {row['Titre_exp']}
Titre VERITE : {row['Titre_verite']}
Score titre : {row['score_titre']}

🔹 LLM1 :
Echantillon EXP : {row['Echantillon_exp1']}
Echantillon VERITE : {row['Echantillon_verite1']}
Verdict : {row['verdict_llm1']}

🔹 LLM2 :
Echantillon EXP : {row['Echantillon_exp2']}
Echantillon VERITE : {row['Echantillon_verite2']}
Verdict : {row['verdict_llm2']}

🔹 LLM3 :
Echantillon EXP : {row['Echantillon_exp3']}
Echantillon VERITE : {row['Echantillon_verite3']}
Verdict : {row['verdict_llm3']}
"""

# === Prompt pour le second LLM ===
prompt = f"""
Tu es un assistant scientifique expert en évaluation d'intelligence artificielle.

On te donne un tableau contenant des comparaisons entre des lignes expérimentales et des lignes de vérité terrain. 
Chaque ligne a été évaluée par trois modèles LLM, qui donnent chacun un verdict sur la correspondance des données (animal, pays, titre, etc.).

Ta tâche est de produire un **bilan d’évaluation des verdicts produits par ces LLMs**.

Tu dois structurer ta réponse en 4 parties :
1. [Statistiques globales] : nombre d’échantillons, pourcentages de verdicts “Oui/Non”, cohérence entre les modèles, score moyen des titres, taux d’accord 2/3 ou 3/3.
2. [Problèmes rencontrés] : erreurs fréquentes, divergences entre les LLMs, types de désaccords (pays, taxonomie, formulation...).
3. [Analyse qualitative] : les verdicts semblent-ils raisonnables ? certains modèles sont-ils plus stricts ou permissifs ?
4. [Suggestions] : comment améliorer les prompts, l’alignement des verdicts ou le système d’évaluation ?

Voici les données à analyser :

{table_lisible}

Merci de structurer clairement ta réponse.
"""

# === Envoi au LLM ===
llm = UniversityLLMAdapter(
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImRjZjZkYWE2LTRjYzMtNDYyOS05MjJiLTkyYzM1NGQzODYwMCJ9.04oTM2nVW8iQTCr8qrs4MknI13dGgqBp85Wq7t2jAeQ",
    api_url="http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions",
    max_tokens=2048,
    temperature=0.2
)

reponse_bilan = llm.generate_answer(prompt)

# === Affichage et sauvegarde ===
print("📊 BILAN DU  LLM :\n")
print(reponse_bilan)

with open("Jugement_dernier.txt", "w", encoding="utf-8") as f:
    f.write(reponse_bilan)
