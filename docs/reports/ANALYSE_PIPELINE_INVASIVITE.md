# Analyse de la Pipeline RAG - Facteurs d'Influence sur la Conclusion d'Invasivité du LLM

## 1. ARCHITECTURE GÉNÉRALE

La pipeline RAG utilise **deux chemins parallèles** pour analyser l'invasivité :
- **HyDE générale** : Pour l'analyse générale des protocoles
- **HyDE invasive_detection** : Spécifiquement pour détecter l'invasivité

---

## 2. FACTEURS INFLUENÇANT LA CONCLUSION D'INVASIVITÉ

### 2.1 **PROMPTS ET TEMPLATES**

#### A. Template HyDE Invasive Detection (Ligne 55-60 pipelines.py)
```
Paramètre clé: invasive_detection=True
```
**Différence majeure avec HyDE générale:**
- Focus explicite sur **"justifie clairement son niveau d'invasivité"**
- Demande de mentionner: **capture d'animaux, anesthésie, stress, perturbation**
- Génère des documents hypothétiques **orientés invasivité** (75-100 mots)
- Promeut un langage scientifique **avec nuance**

**Impact:** Les documents générés avec ce prompt sont biaisés vers l'analyse d'invasivité → influencent l'embedding et la récupération

#### B. Prompt d'Analyse d'Invasivité (prompts.py - INVASIVITY_ANALYSIS_PROMPT)
**Variables clés:**
- `{definition}` : La définition de Taberlet et al. (non-invasif vs invasif)
- `{protocol}` : Le protocole à analyser

**Impact CRITIQUE:**
- La **définition transmise** (Taberlet par défaut) change complètement la classification
- Variation possible dans `mainRag.py` ligne 185 : `--definition`

#### C. Prompt de Détection d'Annonce Non-Invasive (NON_INVASIVE_DETECTION_PROMPT - prompts.py)
**Context:**
- Détecte si les **auteurs affirment** que le protocole est non-invasif
- Peut contredire l'analyse objective

**Logique problématique:**
- Ligne 313-320 : Le prompt reconnaît les "péchés" mais cherche une **annonce d'auteur**
- Peut aboutir à : "Les auteurs le disent non-invasif, mais c'est faux techniquement"

---

### 2.2 **PARAMÈTRES DU LLM (UniversityLLMAdapter)**

#### Temperature (ligne 117, pipelines.py)
```python
temperature=0.7  # Pour HyDE
```

**Impact:**
- `0.7` = modération entre créativité et consistance
- **Valeur basse (0.1-0.2)** = réponses plus déterministes (moins de variation)
- **Valeur haute (0.9)** = plus de variation, risque d'incohérence

**Test recommandé:** Varier entre 0.1 et 0.9 pour voir la stabilité des conclusions

#### Max Tokens
```python
max_tokens=500  # HyDE
max_tokens=1024 # Défaut UniversityLLMAdapter
```

**Impact:**
- Peut tronquer les justifications
- Documents hypothétiques limités (75-100 mots = ~15-20 tokens)

#### System Prompt
```python
system_prompt = "Tu es un assistant scientifique rigoureux."
```

**Impact CRITIQUE:**
- Utilisé dans CHAQUE appel LLM
- Peut influencer la rigueur de la classification

---

### 2.3 **EMBEDDINGS ET RETRIEVAL**

#### Modèle d'Embedding (embedder_model)
- Par défaut : `SentenceTransformersDocumentEmbedder`
- **Impact:** La qualité du modèle affecte la pertinence des documents récupérés

#### Nombre de Générations (n_generations)
```python
"generator": {"n_generations": 3}  # rag_system.py ligne 309
```

**Impact:**
- 3 documents hypothétiques générés par question
- Moyenne de leurs embeddings = query vector final
- Plus de générations = moyenne plus stable (moins de variance)

#### Top-K et Dynamique de Seuil
```python
top_k=8  # mainRag.py ligne 186 (par défaut)
top_k=6  # retrieve_with_hyde() par défaut
```

**Impact:**
- Plus de documents = plus de contexte
- Mais dilution possible avec des documents non pertinents

#### Fonction de Similarité
```python
embedding_similarity_function="cosine"  # rag_system.py ligne 186
```

**Impact:**
- Cosine = angle entre vecteurs (standard)
- Alternative : euclidienne (distance absolue)

---

### 2.4 **DONNÉES D'ENTRÉE**

#### A. Definition de l'Invasivité (mainRag.py - argument --definition)
```python
DEFAULT: "Selon Taberlet et al. (1999), l'échantillonnage d'ADN non invasif 
désigne toute méthode permettant d'obtenir du matériel génétique sans avoir 
à capturer, blesser, ni perturber significativement l'animal..."
```

**Test recommandé:** 
- Changez la définition → change la classification
- Essayez une définition très stricte vs très permissive

#### B. Les "Sept Péchés" (SEVEN_SINS_DEFINITIONS - prompts.py)
```
1. Mauvaise classification des fèces
2. Confusion entre "non-invasif médical" et "non-invasif écologique"
3. Échantillonnage systématique = invasif
4. Négliger le stress animal
5. Équivalence fausse avec procédure médicale
6. Mauvaise évaluation de la capture
7. Ignorer l'impact territorial
```

**Impact CRITIQUE:**
- Chaque "péché" identifié → classification "invasif"
- Une règle stricte identifie davantage de péchés

---

### 2.5 **LOGIQUE D'ANALYSE (RefineRAGSystem)**

#### Niveau 1 : Extraction des Chunks (rag_system.py - ligne 381+)
```python
def build_fusion_prompt(self, analyses: List[Dict]) -> str
```

**Impact:**
- Fusion des analyses de plusieurs chunks
- Logique : "Si fusion dit invasif, peut être corrigé si autres chunks contredisent"
- Risque : Décisions arbitraires basées sur ordre d'analyse

#### Niveau 2 : Analyse d'Invasivité (rag_system.py - ligne 453+)
```python
def refine_analysis(self, question: str, definition: str = "", top_k: int = 4, title: str = "") -> str:
```

**Paramètres critiques:**
- `top_k=4` par défaut → moins de contexte que standard (8)
- `definition=""` par défaut → peut être None ou vide

#### Niveau 3 : Détection d'Annonce Non-Invasive
```python
class RAGNonInvasiveDetection(RAGSystem)
```

**Logique:** "Les auteurs disent-ils que c'est non-invasif?"
- Indépendant de la réalité objective
- Peut créer contradiction : "c'est invasif, mais l'auteur le dit non-invasif"

---

### 2.6 **POST-TRAITEMENT**

#### Parsing JSON (rag_system.py)
```python
def parse_json(response, base_name, invasive_detection=True)
```

**Risques:**
- Réponse du LLM peut ne pas être JSON valide
- Extraction manuelle avec regex `extract_json_blocks()`
- Perte d'information lors du nettoyage

#### Taux de Confiance
```python
"taux_de_confiance": "Taux entre 0 et 100 avec justification"
```

**Impact:**
- Peut être bas mais conclusion quand même générée
- Aucun filtrage basé sur le taux

---

## 3. MATRICE D'INFLUENCE (Ordre de Criticité)

| Facteur | Impact | Facilité Test | Variabilité |
|---------|--------|---------------|-------------|
| **Definition (Taberlet)** | TRÈS HAUT | Facile | Très haute |
| **Prompt HyDE invasion** | TRÈS HAUT | Facile | Haute |
| **Temperature LLM** | HAUT | Facile | Haute |
| **Nombre générations (n_generations)** | HAUT | Facile | Moyenne |
| **Top-K retrieval** | HAUT | Facile | Moyenne |
| **Seven Sins (règles)** | HAUT | Moyen | Haute |
| **System Prompt** | MOYEN | Facile | Moyenne |
| **Embedder Model** | MOYEN | Difficile | Basse |
| **Max Tokens** | MOYEN | Facile | Basse |
| **Parsing JSON** | MOYEN | Difficile | Moyenne |

---

## 4. SCÉNARIOS DE TEST RECOMMANDÉS

### Test 1: Sensibilité à la Définition
```bash
# Définition stricte
--definition "Non-invasif = aucune manipulation, capture, ou stress"

# Définition permissive  
--definition "Non-invasif = pas de blessure permanente"
```

### Test 2: Sensibilité à la Temperature
```python
temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
# Comparer stabilité des conclusions
```

### Test 3: Sensibilité au nombre de générations
```python
n_generations = [1, 3, 5, 10]
# Observer variance du vecteur moyen d'embedding
```

### Test 4: Sensibilité au Top-K
```python
top_k = [2, 4, 6, 8, 10, 15]
# Observer si plus de documents change la conclusion
```

### Test 5: Impact du System Prompt
```python
# Prompt actuel
"Tu es un assistant scientifique rigoureux."

# Prompt neutre
"Tu es un assistant utile."

# Prompt stricte
"Tu es un expert en éthique animale et strictement opposé à l'invasivité."
```

### Test 6: Impact des "7 Péchés"
```
# Tester si retrait d'une règle change la conclusion
- Sans "Mauvaise classification des fèces"
- Sans "Négliger le stress animal"
- Etc.
```

### Test 7: Ordre des Chunks
```
# Même chunks, ordre différent
# Observer si premier chunk a plus d'impact
```

---

## 5. POINTS DE BASCULEMENT CRITIQUES

Ces points peuvent créer des **basculements binaires** (invasif ↔ non-invasif):

1. **Capture vs pas de capture** → Critère décisif
2. **Stress animal mentionné** → Déclencheur "invasif" fort
3. **Anesthésie utilisée** → Marque d'invasivité
4. **Fèces (contexte d'extraction)** → "Péché 1", très sensible
5. **Marquage territorial** → "Péché 7", très sensible

**Impact:** Un LLM peut basculer à cause d'une seule phrase mal interprétée

---

## 6. SOURCES D'INCOHÉRENCE POSSIBLE

1. **HyDE vs Directly Retrieval** : Documents générés peuvent contredire chunks réels
2. **RAGNonInvasiveDetection vs Objective Analysis** : Contradiction entre annonce auteur et réalité
3. **Temperature Variance** : Même question → 3 réponses différentes possibles
4. **Fusion Logic** : Quand plusieurs chunks → quelle règle de fusion?
5. **JSON Parsing** : Mauvaise extraction → perte de nuance

---

## 7. RECOMMANDATIONS POUR TESTS RIGOUREUX

```python
# Structure de test recommandée:

test_cases = {
    "protocol_1": {
        "chunks": [...],
        "expected": "Non-invasif",
        "variations": {
            "temp_low": 0.1,
            "temp_high": 0.9,
            "strict_definition": "...",
            "permissive_definition": "...",
            "top_k": [4, 8, 15]
        }
    }
}

# Pour chaque variation: run LLM 3 fois (non-déterministe)
# Mesurer: cohérence, confidence score, justification
```

---

## CONCLUSION

La conclusion d'invasivité du LLM dépend d'un **écosystème complexe** où:

- **La définition** est le paramètre dominant
- **Le prompt HyDE** biaisant fortement
- **La temperature** créant variance aléatoire
- **Les retrieval parameters** délimitant le contexte
- **Les règles des 7 péchés** en tant que garde-fous

**Aucun paramètre n'est neutre** - tous influencent le résultat final.

