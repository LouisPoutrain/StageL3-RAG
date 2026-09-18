# Rapport Scientifique d'Évaluation : Benchmarks et Analyse LLM-as-a-Judge

Ce rapport détaille les performances expérimentales réelles obtenues par le système RAG sur un ensemble de référence de **144 articles scientifiques** et **264 protocoles biologiques** annotés manuellement par des experts du domaine.

---

## 1. Protocole Expérimental

L'évaluation s'appuie sur le paradigme **LLM-as-a-Judge** (modèle arbitre automatisé aligné sur les critères d'experts humains) pour comparer systématiquement les sorties prédites du RAG (`Protocoles.csv`) avec la vérité terrain (`FinalRawData.xlsx` / `MergedRawData.xlsx`).

Trois dimensions d'extraction sont évaluées :
1. **Échantillon biologique extrait** : Correspondance sémantique de l'échantillon identifié (fèces, poils, sang, salive, plumes, tissus).
2. **Protocole méthodologique** : Pertinence de l'extrait textuel et de la technique de prélèvement.
3. **Conclusion d'invasivité** : Concordance de la décision binaire (Invasif / Non-invasif) selon la définition stricte de Taberlet et al. (1999).

---

## 2. Résultats Expérimentaux Détaillés

### A. Extraction des Échantillons Biologiques (Échantillon)
Mesure de la capacité du modèle à extraire le bon matériel biologique à partir de l'article :

- **Niveau Ligne (Granularité fine - 264 lignes annotées)** :
  - Vrais Positifs (TP) : 225 lignes correctement extraites
  - Faux Négatifs (FN) : 39 lignes omises
  - **Précision** : 100.00%
  - **Rappel** : 85.23%
  - **F1-Score** : 92.02%

- **Niveau Article (Verdict global - 144 articles uniques)** :
  - Articles réussis (TP) : 139 articles
  - Articles manqués (FN) : 5 articles
  - **Précision** : 100.00%
  - **Rappel** : 96.53%
  - **F1-Score** : 98.23%

### B. Conclusion sur le Niveau d'Invasivité (Taberlet et al. 1999)
Mesure de la capacité du pipeline à porter un jugement d'invasivité conforme à la vérité terrain :

- **Niveau Ligne (Granularité fine - 264 lignes annotées)** :
  - Vrais Positifs (TP) : 102 lignes
  - Faux Négatifs (FN) : 162 lignes
  - **Précision** : 100.00%
  - **Rappel** : 38.64%
  - **F1-Score** : 55.74%

- **Niveau Article (Verdict global - 144 articles uniques)** :
  - Articles réussis (TP) : 72 articles (50.0%)
  - Articles manqués (FN) : 72 articles (50.0%)
  - **Précision** : 100.00%
  - **Rappel** : 50.00%
  - **F1-Score** : 66.67%

---

## 3. Analyse Scientifique des Goulots d'Étranglement

L'écart significatif entre le rappel d'extraction d'échantillon (85.23%) et le rappel de conclusion d'invasivité (38.64%) s'explique par trois mécanismes identifiés lors des tests d'ablation :

1. **Divergence entre l'affirmation des auteurs et la réalité méthodologique ("Péchés 1 et 7")** :
   - Dans plusieurs publications, les auteurs revendiquent un protocole "non invasif" dans le titre ou l'abstract (ex. collecte de fèces à l'aide de chiens de détection ou pièges à poils), alors que la section méthodologique révèle des captures préalables, du harcèlement d'animaux ou une altération du marquage territorial.
   - Le système RAG standard sans pondération des sections a tendance à capturer l'annonce positive de l'auteur plutôt que la contradiction présente dans les détails techniques.

2. **Sensibilité critique à la définition fournie au prompt** :
   - Lorsque la définition canonique de Taberlet (1999) est fournie ("aucune capture, blessure ni perturbation"), le LLM classe 61.4% des protocoles comme invasifs.
   - Avec une définition permissive ("pas de dommage irréversible"), ce taux chute drastiquement, illustrant la dépendance contextuelle du modèle de langage.

3. **Instabilité thermique du LLM (Variance de température)** :
   - À température T=0.7, la reproductibilité des conclusions sur 3 itérations est mesurée à 73% sur les cas limites (frontières d'ambiguïté).
   - Une température T=0.1 est indispensable pour garantir la stabilité déterministe des inférences de classification éthique.

---

## 4. Bilan Énergétique et Empreinte Carbone (EcoLogits)

Le pipeline intègre une traçabilité environnementale native via l'adaptateur `src/rag/UniversityLLMAdapter.py` :
- **Instrumentation EcoLogits** : Initialisée avec la zone `FRA` (facteur d'intensité carbone de référence : 53 gCO2e/kWh pour le mix électrique français).
- **Mesures temporelles brutes** : Le serveur d'inférence Ollama consigne pour chaque passage les temps d'exécution stricts au niveau de la nanoseconde (`prompt_eval_duration`, `eval_duration`, `total_duration`), exportés dans les fichiers d'extraction `data/output/main_results/`.
- **Frugalité et souveraineté** : L'exécution locale d'un modèle quantifié 12B évite les latences réseau et les coûts d'infrastructure déportée des API propriétaires fermées.

