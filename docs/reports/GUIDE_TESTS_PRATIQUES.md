# Guide de Tests Pratiques - Conclusions sur l'Invasivité

## Vue d'Ensemble des Tests

Voici comment structurer tes tests pour identifier quels paramètres influencent vraiment la conclusion:

---

## TEST 1️⃣: DÉFINITION (Impact: **TRÈS HAUT**)

### Protocole de Test
```bash
# Même chunk, définitions différentes

CHUNK: "Les fèces ont été collectées passivement sans capturer les animaux"

# Test 1a: Taberlet Strict
Definition: "Sans capturer, blesser, ni perturber"
Expected Impact: → "Non-invasif"

# Test 1b: Taberlet Permissive  
Definition: "Pas de blessure permanente"
Expected Impact: → "Non-invasif"

# Test 1c: Medical Definition (Piège)
Definition: "Pas de pénétration cutanée"
Expected Impact: → "Non-invasif"

# Test 1d: Strict Ethics
Definition: "Aucun contact, aucune manipulation"
Expected Impact: → "Invasif" ⚠️ (passive collection peut être invasive si stressante)
```

### Matrice Résultats Expected
| Definition | Fèces | Sang | Poils | Capture |
|-----------|-------|------|-------|---------|
| **Taberlet** | Non-inv | Invasif | Non-inv | Invasif |
| **Strict Ethics** | ? | Invasif | Non-inv? | Invasif |
| **Medical** | Non-inv | Non-inv | Non-inv | ? |

**Risque Critique:** Si définition change → résultat change complètement ⚠️

---

## TEST 2️⃣: TEMPÉRATURE (Impact: **HAUT**)

### Protocole de Test
```bash
# Même prompt, même chunks, 3 runs par temperature

CHUNK: "L'animal a été capturé avec un filet, puis relâché"

Temperatures: 0.1, 0.3, 0.5, 0.7, 0.9

Pour CHAQUE temperature:
  Run 1: Qu'est-ce que le LLM dit?
  Run 2: Répétition → même réponse?
  Run 3: Répétition → même réponse?
```

### Ce qu'il faut observer

**Temperature 0.1 (Déterministe):**
- Run 1: "C'est invasif car capture"
- Run 2: "C'est invasif car capture" ✓ Cohérent
- Run 3: "C'est invasif car capture" ✓ Cohérent

**Temperature 0.9 (Créatif):**
- Run 1: "C'est invasif car capture + stress"
- Run 2: "C'est non-invasif car relâché vivant" ❌ DIVERGENCE
- Run 3: "C'est minimalement invasif, classification ambiguë" ❌ DIVERGENCE

**Interprétation:** Haute temperature = variance aléatoire, basse = stabilité

---

## TEST 3️⃣: N_GENERATIONS (Impact: **MOYEN-HAUT**)

### Protocole de Test
```bash
# Hypothetical Document Embeddings: combien de documents générés?

n_generations = [1, 3, 5, 10]

Pour CHAQUE valeur:
  1. Générer N documents hypothétiques
  2. Comparer leurs embeddings
  3. Vecteur moyen = query vector pour retrieval
  4. Mesurer variance du vecteur moyen

# Question: Plus de génération = plus stable?
# Si 1 doc génère "capture" → classification invasif
# Si 3 docs génèrent [capture, passive, handling] → moyenne plus neutre
```

### Ce qu'il faut mesurer
```
Metric: Variance du vecteur d'embedding moyen

n_gen=1: High variance (1 seul doc = pas de moyenne)
n_gen=3: Medium variance (3 docs → moyenne assez stable)
n_gen=5: Lower variance
n_gen=10: Even more stable

→ Plus de générations = embedding query plus stable = retrieval plus consistant
```

---

## TEST 4️⃣: TOP-K RETRIEVAL (Impact: **HAUT**)

### Protocole de Test
```bash
# Combien de documents récupérés du corpus?

top_k = [2, 4, 6, 8, 10, 15]

Pour CHAQUE top_k:
  1. Récupérer top_k documents du corpus
  2. Analyser invasivité avec ces documents
  3. Comparer avec top_k précédent

EXEMPLE:
top_k=2: [Doc_captureFish, Doc_passiveFeathers]
         → LLM voit mixture capture+passive
         → Peut biaiser vers moyenne

top_k=8: [Doc_captureFish, Doc_passiveFeathers, Doc_anesthesia, ...]
         → Plus de contexte
         → Peut nuancer davantage
```

### Risques
```
top_k TROP BAS (2-4):
  - Sampling bias possible
  - Documents peu pertinents ignorés
  - Conclusion peut basculer

top_k TROP HAUT (15+):
  - Documents non pertinents inclus → bruit
  - LLM peut se concentrer sur détails non pertinents
  - Variance augmente
```

---

## TEST 5️⃣: SYSTEM PROMPT (Impact: **MOYEN**)

### Protocole de Test
```bash
# Quatre system prompts, même chunks

CHUNK: "Nous avons prélevé du sang des animaux en captivité avec consentement"

# Prompt A: Scientific (Actuel)
System: "Tu es un assistant scientifique rigoureux"
Expected: "Analyse objective de l'invasivité"

# Prompt B: Neutral
System: "Tu es un assistant utile et neutre"
Expected: "Analyse sans biais déclaré"

# Prompt C: Strict Ethics
System: "Tu es expert en éthique animale, opposé à l'invasivité"
Expected: "Tendance à classifier INVASIF ⚠️"

# Prompt D: Pragmatic
System: "Tu valorises la recherche, capture acceptable si justifiée"
Expected: "Tendance à classifier NON-INVASIF ⚠️"

RÉSULTAT ATTENDU:
  A & B: "Invasif" (prélèvement de sang = invasif)
  C: "Très invasif" (aucune justification acceptable)
  D: "Minimalement invasif" (capture justifiée par recherche)
```

**Interprétation:** Le system prompt crée du BIAIS, pas une neutralité garantie

---

## TEST 6️⃣: LES 7 PÉCHÉS (Impact: **HAUT**)

### Protocole de Test
```bash
# Chaque "péché" = règle stricte. La détecter change-t-elle la conclusion?

CHUNK: "Nous avons collecté des fèces de mandrilles via capture au filet"

# Sin 1: Mauvaise classification des fèces
→ Détecté: "Fèces mais capture" → Peut être invasif si stress
Result: Inversé de "Non-invasif" à "INVASIF"

# Sin 3: Échantillonnage systématique
→ Détecté: "Quotidiennement pendant 6 mois = systématique"
Result: Changé de "Non-invasif" à "INVASIF"

# Sin 4: Négliger le stress
→ Détecté: "Pas mentionné comment stress minimal"
Result: Peut inverser la conclusion

# Sin 7: Ignorer impact territorial
→ Détecté: "Marquage territorial mentionné"
Result: Peut inverser de "Non-invasif" à "INVASIF"
```

### Test d'Impact
```bash
# Pour chaque péché, tester:

1. AVEC le péché activé
   → Conclusion: "Invasif"

2. SANS le péché (désactiver la règle)
   → Conclusion: "Non-invasif" ✓ Péché actif a un impact

OU

   → Conclusion: "Invasif" ✗ Péché n'a pas d'impact
```

---

## TEST 7️⃣: ORDRE DES CHUNKS (Impact: **MOYEN**)

### Protocole de Test
```bash
# L'ordre des chunks affecte-t-il la conclusion?

CHUNKS_ORIGINAL:
  [1] "Les animaux ont été capturés"
  [2] "Avec anesthésie légère"
  [3] "Puis relâchés immédiatement"

RÉSULTAT: "Invasif (capture + anesthésie)"

CHUNKS_REORDERED_1:
  [1] "Puis relâchés immédiatement"
  [2] "Les animaux ont été capturés"
  [3] "Avec anesthésie légère"

RÉSULTAT: Même conclusion? Ou changé?

CHUNKS_REORDERED_2:
  [1] "Avec anesthésie légère"
  [2] "Puis relâchés immédiatement"
  [3] "Les animaux ont été capturés"

RÉSULTAT: Même conclusion? Ou changé?
```

### Ce qu'il faut observer
```
Hypothèse 1: Recency Bias
→ Dernier chunk a plus d'impact
→ Si "relâchés immédiatement" est dernier → "Non-invasif"?

Hypothèse 2: Primacy Bias
→ Premier chunk a plus d'impact
→ Si "capturés" est premier → "Invasif"?

Hypothèse 3: No Bias
→ Ordre ne change rien
→ LLM comprend contexte global
```

---

## 🔬 STRUCTURE DE TEST RECOMMANDÉE

### Pour chaque test:

```python
test_result = {
    "test_name": "...",
    "protocol": "...",
    "parameter": {
        "name": "...",
        "values_tested": [...]
    },
    "results": [
        {
            "value": 0.7,
            "run_1": "Invasif",
            "run_2": "Invasif",
            "run_3": "Invasif",
            "consistency": "100%",
            "justification": "Capture d'animal = invasif"
        },
        ...
    ],
    "conclusion": "Parameter IMPACTFUL / NOT IMPACTFUL"
}
```

### Format JSON à générer
```json
{
  "test_suite": "invasivity_factors",
  "tests": [
    {
      "test_id": 1,
      "name": "Definition Sensitivity",
      "protocol_tested": "Fecal sampling from captured animals",
      "results": [
        {
          "definition": "taberlet_strict",
          "conclusion": "Invasif",
          "confidence": 95,
          "justification": "Capture = invasif selon Taberlet"
        },
        {
          "definition": "medical_non_invasive",
          "conclusion": "Non-invasif",
          "confidence": 70,
          "justification": "Pas de pénétration cutanée"
        }
      ],
      "parameter_impactful": true,
      "variance_across_parameter": "VERY HIGH"
    }
  ],
  "impact_ranking": [
    "Definition (VERY HIGH)",
    "HyDE_Prompt (HIGH)",
    "Temperature (HIGH)",
    "Top_K (HIGH)",
    "Seven_Sins (MEDIUM-HIGH)",
    "System_Prompt (MEDIUM)",
    "N_Generations (MEDIUM)",
    "Chunk_Order (LOW-MEDIUM)"
  ]
}
```

---

## 📊 TABLEAU DE SYNTHÈSE

```
┌─────────────────┬──────────┬───────────┬──────────────┬──────────────┐
│ Facteur         │ Impact   │ Variance  │ Testabilité  │ Criticalité  │
├─────────────────┼──────────┼───────────┼──────────────┼──────────────┤
│ Definition      │ TRÈS HT  │ Très haute│ Très facile  │ 🔴 CRITIQUE  │
│ HyDE Prompt     │ TRÈS HT  │ Haute     │ Moyen        │ 🔴 CRITIQUE  │
│ Temperature     │ HAUT     │ Très haute│ Très facile  │ 🟠 IMPORTANT │
│ Top-K           │ HAUT     │ Moyenne   │ Facile       │ 🟠 IMPORTANT │
│ Seven Sins      │ HAUT     │ Haute     │ Difficile    │ 🟠 IMPORTANT │
│ System Prompt   │ MOYEN    │ Moyenne   │ Facile       │ 🟡 MODÉRÉ    │
│ N_Generations   │ MOYEN    │ Moyenne   │ Moyen        │ 🟡 MODÉRÉ    │
│ Chunk Order     │ MOYEN    │ Basse     │ Difficile    │ 🟢 FAIBLE    │
│ Embedder Model  │ MOYEN    │ Basse     │ Très difficile│ 🟢 FAIBLE    │
└─────────────────┴──────────┴───────────┴──────────────┴──────────────┘
```

---

## 🎯 PROCHAINES ÉTAPES

1. **Exécuter Test 1 (Definition)** → voir l'impact énorme
2. **Exécuter Test 2 (Temperature)** → voir la variance
3. **Exécuter Test 4 (Top-K)** → paramètre de retrieval
4. **Exécuter Test 5 (System Prompt)** → biais détecté?
5. **Combiner résultats** → Matrice d'influence

---

## ⚠️ RÉSULTATS À ATTENDRE

### Probabilité élevée:
- **Definition** change complètement le résultat (invasif ↔ non-invasif)
- **Temperature** crée variance aléatoire
- **Top-K** peut diluer ou concentrer les conclusions

### Probabilité moyenne:
- **HyDE Prompt** biaise légèrement (plus difficile à tester)
- **System Prompt** crée biais déclaré

### Probabilité basse:
- **Chunk Order** n'affecte pas le résultat final

