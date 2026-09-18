# Guide d'utilisation - Tests Comparatifs de la Pipeline RAG

## 🎯 Objectif

Ce script exécute la **vraie pipeline RAG** avec différents paramètres et **compare les résultats entre eux** pour identifier quels facteurs influencent réellement les conclusions sur l'invasivité.

## 📁 Fichiers Créés

### Scripts Principaux
- `test_parameter_influence.py` - Script pour exécuter les tests comparatifs
- `analyze_articles.py` - Script pour analyser MergedRawData.xlsx

### Documents d'Analyse
- `ANALYSE_PIPELINE_INVASIVITE.md` - Analyse complète des facteurs d'influence
- `GUIDE_TESTS_PRATIQUES.md` - Guide pratique avec protocoles de test
- `ARTICLES_TESTS_RECOMMANDES.md` - 5 articles sélectionnés pour les tests

### Résultats (générés automatiquement)
- `test_results_comparative/` - Tous les résultats JSON des tests

## 🚀 Comment Lancer les Tests

### Test 1: Impact de la DÉFINITION
```bash
python3 test_parameter_influence.py --article 012017-jfwm-007 --test definition
```

**Ce test fait:**
- Exécute la pipeline RAG avec 5 définitions différentes
- Compare les protocoles détectés et leurs évaluations
- Génère un tableau comparatif montrant les divergences

**Durée estimée:** 10-15 minutes (5 définitions × ~2-3 min chacune)

**Fichiers générés:**
- `test_results_comparative/definition_test_012017-jfwm-007.json`
- `test_results_comparative/definition_analysis_012017-jfwm-007.json`

### Test 2: Impact de la TEMPÉRATURE
```bash
python3 test_parameter_influence.py --article 012017-jfwm-007 --test temperature
```

**Ce test fait:**
- Exécute avec 5 températures (0.1, 0.3, 0.5, 0.7, 0.9)
- 3 runs par température pour mesurer la variance
- Calcule la cohérence des résultats

**Durée estimée:** 30-45 minutes (5 temp × 3 runs × ~2-3 min)

**Fichiers générés:**
- `test_results_comparative/temperature_test_012017-jfwm-007.json`
- `test_results_comparative/temperature_analysis_012017-jfwm-007.json`

### Test 3: Impact du TOP-K
```bash
python3 test_parameter_influence.py --article 012017-jfwm-007 --test topk
```

**Ce test fait:**
- Exécute avec 6 valeurs de top-k (2, 4, 6, 8, 10, 15)
- Compare comment le nombre de documents récupérés affecte la conclusion

**Durée estimée:** 12-18 minutes

### Test 4: Impact du N_GENERATIONS
```bash
python3 test_parameter_influence.py --article 012017-jfwm-007 --test ngen
```

**Ce test fait:**
- Exécute avec 4 valeurs (1, 3, 5, 10 générations HyDE)
- Mesure l'impact du nombre de documents hypothétiques

**Durée estimée:** 8-12 minutes

### Test ALL: Tous les Tests
```bash
python3 test_parameter_influence.py --article 012017-jfwm-007 --test all
```

**Durée estimée:** 1-1.5 heures

## 📊 Interpréter les Résultats

### Format des Fichiers JSON

Chaque test génère 2 fichiers:

#### 1. Fichier de test brut (ex: `definition_test_*.json`)
```json
[
  {
    "test_id": "taberlet_original_temp0.7_k8",
    "success": true,
    "protocols": [
      {
        "protocole": "Échantillonnage d'ADN à partir de guano",
        "evaluation_invasivite": "Invasif",
        "taux_de_confiance": "80%"
      }
    ],
    "parameters": {
      "definition_key": "taberlet_original",
      "temperature": 0.7,
      "top_k": 8
    }
  }
]
```

#### 2. Fichier d'analyse (ex: `definition_analysis_*.json`)
```json
[
  {
    "definition": "taberlet_original",
    "num_protocols": 2,
    "invasif_count": 2,
    "non_invasif_count": 0
  },
  {
    "definition": "taberlet_permissive",
    "num_protocols": 2,
    "invasif_count": 1,
    "non_invasif_count": 1
  }
]
```

### Exemple de Sortie Console

```
====================================================================================================
ANALYSE: IMPACT DE LA DÉFINITION
====================================================================================================

📊 Tableau Comparatif:
Définition                Protocoles   Invasif    Non-invasif    
----------------------------------------------------------------------
taberlet_original         2            2          0
taberlet_strict           2            2          0
taberlet_permissive       2            1          1          ← DIVERGENCE
medical_definition        2            0          2          ← DIVERGENCE
minimal_definition        2            1          1          ← DIVERGENCE

🔍 Divergences Détectées:
  • taberlet_permissive vs taberlet_original: -1 protocoles invasifs
  • medical_definition vs taberlet_original: -2 protocoles invasifs
  • minimal_definition vs taberlet_original: -1 protocoles invasifs
```

## 🔍 Analyser les Divergences

### Cas 1: Définition Change la Conclusion

Si vous voyez:
```
taberlet_original: 2 invasif, 0 non-invasif
medical_definition: 0 invasif, 2 non-invasif
```

**Interprétation:** La définition a un impact **CRITIQUE**. Elle inverse complètement la conclusion.

### Cas 2: Température Crée de la Variance

Si vous voyez:
```
Temp    Runs   Cohérence    Invasif (min-max)
0.1     3      100%         2-2               ← Stable
0.7     3      67%          1-2               ← Variance moyenne
0.9     3      33%          0-2               ← Haute variance!
```

**Interprétation:** À température élevée (0.9), le LLM est **instable**. Même input → sorties différentes.

### Cas 3: Top-K Affecte le Résultat

Si vous voyez:
```
Top-K    Protocoles    Invasif
2        2             2
4        2             2
8        2             1        ← Change ici!
15       2             1
```

**Interprétation:** Plus de documents (top-k=8+) → conclusion change. Peut-être que des documents contradictoires sont récupérés.

## ⚙️ Configuration Avancée

### Changer les Articles Testés

Éditez `test_parameter_influence.py` ligne ~30:
```python
SELECTED_ARTICLES = {
    "012017-jfwm-007": {
        "json_file": "data/input/012017-jfwm-007.json",
        "title": "..."
    },
    # Ajoutez vos articles ici
}
```

### Ajouter des Définitions

Éditez `test_parameter_influence.py` ligne ~48:
```python
DEFINITIONS_TO_TEST = {
    "ma_definition_custom": "Votre définition ici...",
    ...
}
```

### Modifier les Valeurs de Test

Dans les fonctions de test, changez les listes:
```python
temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]  # Ligne ~248
top_k_values = [2, 4, 6, 8, 10, 15]        # Ligne ~291
n_gen_values = [1, 3, 5, 10]                # Ligne ~332
```

## 🐛 Dépannage

### Erreur: "No module named 'rag'"
```bash
# Vérifiez que vous êtes dans le bon répertoire
cd /path/to/StageL3-RAG
source .venv/bin/activate
```

### Erreur: "'list' object has no attribute 'get'"
Le fichier JSON d'entrée est mal formaté. Vérifiez qu'il contient bien un dict avec les clés attendues.

### Le test est très lent
C'est normal ! La pipeline RAG exécute:
- Embedding des chunks
- Génération HyDE (3 documents)
- Retrieval 
- Analyse LLM (plusieurs appels)

Chaque run prend 2-3 minutes. Soyez patient.

### Résultats vides
Vérifiez que:
1. Le fichier JSON existe dans `data/input/`
2. Les chunks sont indexés dans le document store
3. Le LLM répond correctement (pas d'erreur API)

## 📈 Prochaines Étapes

Après avoir les résultats des tests:

1. **Analyser les JSON générés** dans `test_results_comparative/`
2. **Identifier les facteurs critiques** (ceux qui causent le plus de divergences)
3. **Créer un rapport** consolidé avec les insights
4. **Recommandations** pour stabiliser la pipeline

## 💡 Exemples d'Insights à Chercher

### Question 1: La définition est-elle le facteur dominant?
Regardez si changer la définition inverse les conclusions (invasif ↔ non-invasif).

### Question 2: Le LLM est-il stable?
À température 0.1, le LLM devrait donner 100% de cohérence. Sinon, problème de déterminisme.

### Question 3: Top-k optimal?
Y a-t-il une valeur de top-k où les résultats se stabilisent?

### Question 4: N_generations utile?
Est-ce que 10 générations HyDE donnent de meilleurs résultats que 1?

## 📞 Support

En cas de problème, vérifiez:
- Les logs dans `test_results_comparative/`
- La sortie console complète
- Les fichiers JSON générés

---

**Temps total estimé pour tous les tests:** 1-1.5 heures par article
**Nombre d'articles recommandés:** 3-5
**Durée totale du projet de tests:** 3-7 heures
