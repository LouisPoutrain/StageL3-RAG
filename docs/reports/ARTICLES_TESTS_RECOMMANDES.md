# 5 Articles Sélectionnés pour les Tests de la Pipeline RAG

## Résumé Exécutif

Basé sur l'analyse de 379 protocoles dans 276 articles uniques du fichier MergedRawData.xlsx, voici les **5 articles recommandés** pour tester la robustesse et la sensibilité de la pipeline d'analyse d'invasivité du LLM.

Chaque article a été sélectionné pour tester un aspect critique de la pipeline:

---

## 📋 ARTICLE #1: CONTRADICTION TITRE/ÉVALUATION

**Filename:** `012017-jfwm-007`

**Titre:** "Identification of Southeastern Bat Species Using Noninvasive Genetic Sampling of Individual Guano Pellets"

**Auteurs:** Veronica A Brown, Emma V Willcox, Kirstin E Fagan, Riley F Bernard

**Année:** 2017

### Pourquoi intéressant ?

**Contradiction majeure:**
- ✅ Titre dit explicitement: **"Noninvasive"**
- ❌ Évaluation LLM: **"Invasif"**

### Protocoles analysés (2 dans cet article)

| Protocole | Sample | Évaluation | Confiance | Justification |
|-----------|--------|-----------|-----------|--------------|
| Échantillonnage d'ADN à partir de **guano et de tissu** | Guano + Tissu | **Invasif** | 100% | La capture d'animaux vivants pour obtenir du tissu = invasif |
| Échantillonnage d'ADN à partir de **guano** | Fèces | **Invasif** | 80% | Collecte systématique et exhaustive peut affecter le comportement |

### Péchés détectés: [1, 3]
- **Péché #1:** Mauvaise classification des fèces (collecte opportuniste vs systématique)
- **Péché #3:** Échantillonnage systématique = invasif

### Excerpts clés
```
"We used DNA extracted from tissue and guano samples..."
"Guano samples consisted of 1-5 pellets collected from live-captured bats."
"Tissue samples varied from 2-to 3-mm wing biopsies from live individuals..."
```

### Test à effectuer
```
✓ Teste la DÉFINITION
  - Le LLM change-t-il sa conclusion si on varie la définition de Taberlet?
  - Comment gère-t-il la contradiction titre/contenu?
  - Le prompt HyDE "invasive_detection" change-t-il le résultat?
```

---

## ⚠️ ARTICLE #2: BASSE CONFIANCE

**Filename:** `1-s2.0-S0006320713002772-main`

**Titre:** "Combining camera-trapping and noninvasive genetic data in a spatial capture-recapture framework to improve density estimates of elusive wildlife"

**Année:** 2013

### Pourquoi intéressant ?

**Confiance basse (75%):**
- Le LLM classe le protocole comme **"Invasif"**
- Mais avec seulement **75% de confiance** (vs 100% ou 80% usuels)
- Indique une **ambiguïté ou incertitude** du modèle

### Protocole analysé

| Protocole | Évaluation | Confiance | Justification |
|-----------|-----------|-----------|--------------|
| Utilisation de **pièges photographiques** | Invasif | **75%** | Manque d'informations précises sur l'impact des pièges |

### Raison de la basse confiance
```
La méthodologie est partiellement détaillée, mais il manque:
- L'impact exact des pièges photographiques sur le comportement
- Si les animaux sont attirants/répulsifs
- La fréquence d'activation des pièges
```

### Test à effectuer
```
✓ Teste la TEMPÉRATURE ET VARIANCE
  - Avec temperature 0.1: LLM maintient 75% ou change?
  - Avec temperature 0.9: Variance augmente?
  - Le taux de confiance est-il stable ou fluctue?
  
✓ Teste la STABILITÉ
  - Exécuter 5 fois avec mêmes paramètres
  - Observer si conclusion change malgré basse confiance
```

---

## ❓ ARTICLE #3: MANQUE D'INFORMATIONS

**Note:** La recherche a montré **0 protocoles** avec "Manque d'informations"

**Solution:** Utiliser l'article #1 (Bats) qui a **deux évaluations** pour une même espèce:

**Filename:** `012017-jfwm-007` (déjà décrit ci-dessus)

**Protocole substitut pour ce critère:**

| Protocole | Évaluation |
|-----------|-----------|
| Guano + Tissu | **Invasif** (clair: capture) |
| Guano seul | **Invasif** (ambigu: pourquoi?) |

### Test à effectuer
```
✓ Teste l'INTERPRÉTATION DE L'AMBIGUÏTÉ
  - Guano = comment le LLM le classe?
  - Le contexte de la capture affecte-t-il la classification du guano seul?
  - Sensibilité au contexte vs classification isolée?
```

---

## ☠️ ARTICLE #4: PÉCHÉS IDENTIFIÉS

**Filename:** `012017-jfwm-007`

**Titre:** (Même article que #1 - efficience)

### Péchés détectés: **[1, 3]**

#### Péché #1: Mauvaise classification des fèces
```
Problème: Les fèces sont collectées à partir de chauves-souris capturées vivantes.
Le LLM devrait détecter: "fèces MAIS capture" = invasif
Pas juste: "fèces" = non-invasif (piège classique)
```

#### Péché #3: Échantillonnage systématique
```
Problème: "collected from each accumulation" = exhaustif/systématique
Le LLM devrait détecter: échantillonnage systématique peut perturber
Règle: Si systématique ET affecte comportement = invasif
```

### Test à effectuer
```
✓ Teste les RÈGLES DES 7 PÉCHÉS
  - Désactiver péché #1: conclusion change?
  - Désactiver péché #3: conclusion change?
  - Si on retire TOUS les péchés: classification devient non-invasif?
  
✓ Teste la RIGUEUR DU LLM
  - Le LLM identifie-t-il toujours les péchés?
  - Ou seulement si le prompt les mentionne explicitement?
```

---

## 🔀 ARTICLE #5: VARIANCE INTRA-ARTICLE

**Filename:** `Genetic structure and population history of wintering Asian Great Bustard (Otis tarda dybowskii) in`

### Pourquoi intéressant ?

**Même article, 4 protocoles DIFFÉRENTS, 3 évaluations DIFFÉRENTES:**

| Protocole | Évaluation |
|-----------|-----------|
| Fecal sampling of **Great Bustard** | ✅ **Non invasif** |
| Tissue sampling of **Leopard Cat** | ❌ **Invasif** |
| Stomach sampling of **Leopard Cat** | ❌ **Invasif** |
| Scat sampling of **Leopard Cat** | ❌ **Invasif - Territory marking** |

### Observation clé
```
Même type d'échantillon (scats/guano):
  - Great Bustard = NON-INVASIF
  - Leopard Cat = INVASIF (+ territory marking!)
  
Pourquoi la différence?
→ Espèce différente → critères différents?
→ Contexte d'extraction différent?
→ Le LLM applique-t-il des règles taxon-spécifiques?
```

### Test à effectuer
```
✓ Teste la COHÉRENCE INTRA-ARTICLE
  - Comment le LLM gère-t-il plusieurs protocoles du MÊME article?
  - Biais vers le premier protocole (primacy)?
  - Biais vers le dernier protocole (recency)?
  - Compréhension du contexte global?
  
✓ Teste l'ORDRE DES CHUNKS (récupération)
  - Reorder les chunks: conclusion change?
  - Great Bustard en premier vs en dernier
  
✓ Teste la SENSIBILITÉ TAXON
  - Le LLM sait-il que différentes espèces = différents critères?
  - "Territory marking" pour Leopard Cat = critère détecté correctement?
```

---

## 📊 Tableau Récapitulatif

| # | Filename | Article | Critère Testé | Résultat Attendu |
|---|----------|---------|----------------|-----------------|
| 1 | `012017-jfwm-007` | Bats - Guano & Tissue | CONTRADICTION | Titre=Non-inv, Eval=Invasif |
| 2 | `1-s2.0-S0006320713002772-main` | Camera traps | BASSE CONFIANCE | 75% confiance |
| 3 | `012017-jfwm-007` | (Même #1) | AMBIGUÏTÉ | Guano seul vs + capture |
| 4 | `012017-jfwm-007` | (Même #1) | PÉCHÉS [1,3] | Mauvaise classe fèces + systématique |
| 5 | `Asian Great Bustard` | Multi-espèces | VARIANCE INTRA | Non-inv (Great Bustard) vs Invasif (Leopard Cat) |

---

## 🎯 Plan de Tests Recommandé

### Phase 1: Test de base (24h)
```bash
# Pour chaque article, tester avec configuration par défaut
for article in [1, 2, 4, 5]:
  - Analyser avec RAG standard
  - Enregistrer résultat (invasif/non-invasif + confiance + justification)
  - Comparer avec MergedRawData.xlsx
```

### Phase 2: Test de sensibilité (48h)
```bash
# Article #1 (Contradiction)
- Tester avec 5 définitions différentes
- Observer si conclusion change (invasif ↔ non-invasif)

# Article #2 (Basse confiance)
- Tester avec temperature [0.1, 0.3, 0.5, 0.7, 0.9]
- Exécuter 3 fois chaque temperature
- Mesurer variance

# Article #5 (Variance)
- Reorder chunks (Great Bustard premier vs dernier)
- Exécuter avec différents top-k values
```

### Phase 3: Test de la pipeline (48h)
```bash
# Comparaison des facteurs d'influence
- Quel facteur a le plus d'impact sur la conclusion?
- Classement final: Definition > Prompt > Temperature > Top-K > ...
```

---

## 📁 Emplacements des Articles

```
Data/input/012017-jfwm-007.json
Résultats/MergedRawData.xlsx (rows 0-1 pour article #1)
```

### Accès au contenu complet
```bash
# Récupérer les chunks pour article #1
grep -l "012017-jfwm-007" data/chunks/**/*.txt

# Voir les métadonnées complet dans Excel
Résultats/MergedRawData.xlsx:Sheet1:Row 0-1 (pour article #1)
```

---

## ⏱️ Effort Estimé

| Article | Test | Durée | Priorité |
|---------|------|-------|----------|
| #1 | Définition x5 | 30 min | 🔴 HAUTE |
| #2 | Temperature x5 | 45 min | 🔴 HAUTE |
| #4 | Péchés (6 tests) | 1h | 🟠 MOYEN |
| #5 | Ordre chunks | 1h | 🟠 MOYEN |

**Total:** ~3-4 heures pour tous les tests de sensibilité

---

## 💡 Insights Clés

1. **Article #1 est RÉFÉRENCE** pour 60% des tests (apparaît 3x)
   → Contradiction titre/évaluation + péchés + ambiguïté

2. **Article #2 teste l'incertitude** → LLM communique bien sa confiance?

3. **Article #5 teste la cohérence** → même article, différentes conclusions

4. **Aucun article avec "Manque d'informations"** → le LLM force toujours une conclusion?

5. **17 articles avec CONTRADICTION** → Biais possible vers "Non-invasif" si titre dit ça?

