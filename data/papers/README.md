# Catalogue des Articles Scientifiques et Corpus de Référence

Ce document répertorie les articles scientifiques et publications fondamentales utilisés pour l'entraînement, l'évaluation et la validation du pipeline RAG d'analyse d'invasivité génétique.

---

## 1. Publications Fondatrices

### Noninvasive Genetic Sampling: Look Before You Leap
- **Auteurs** : Pierre Taberlet, Lisette P. Waits, Gordon Luikart
- **Année** : 1999
- **Revue** : *Trends in Ecology & Evolution* (TREE), Vol. 14, No. 8, pp. 323–327
- **DOI** : [10.1016/S0169-5347(99)01637-7](https://doi.org/10.1016/S0169-5347(99)01637-7)
- **Rôle dans le projet** : Définition théorique canonique de l'échantillonnage non-invasif (obtention de matériel génétique sans capture, sans blessure et sans perturbation significative de l'animal cible). Règle de décision centrale du système RAG.

### Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)
- **Auteurs** : Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
- **Année** : 2022
- **Dépôt** : arXiv:2212.10496 [cs.IR]
- **Lien direct arXiv** : [https://arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496)
- **OpenReview / ACL** : [https://openreview.net/forum?id=gJz5-pBRsE](https://openreview.net/forum?id=gJz5-pBRsE)
- **Rôle dans le projet** : Algorithme de génération de documents hypothétiques pour combler l'écart sémantique entre les questions utilisateur courtes et les sections denses de méthodologie biologique.

---

## 2. Articles du Corpus Local

### Carrion fly-derived DNA as a tool for comprehensive and cost-effective assessment of mammalian biodiversity
- **Fichier local** : `Molecular Ecology - 2013 - Calvignac‐Spencer - Carrion fly‐derived DNA as a tool for comprehensive and cost‐effective.pdf`
- **Auteurs** : Sébastien Calvignac-Spencer, Klaus Merkel, Nicole Kutzner, Hjalmar S. Kühl, Christophe Boesch, Peter M. Kappeler, Sonja Metzger, Grit Schubert, Fabian H. Leendertz
- **Année** : 2013
- **Revue** : *Molecular Ecology*, Vol. 22, Issue 4, pp. 915–924
- **DOI** : [10.1111/mec.12183](https://doi.org/10.1111/mec.12183)
- **PubMed** : [PMID: 23301777](https://pubmed.ncbi.nlm.nih.gov/23301777/)
- **Résumé technique** : Démonstration de l'utilisation de l'ADN d'invertébrés hématophages/nécrophages (iDNA) comme méthode d'échantillonnage de la faune mammalienne sans contact direct avec les animaux vivants. Cas d'école pour la détection du caractère non-invasif d'une collecte biologique indirecte.

---

## 3. Articles Clés du Benchmark (144 Articles Évalués)

Les métadonnées complètes des 144 articles analysés par le système RAG et annotés par des biologistes sont conservées dans `data/benchmarks/FinalRawData.xlsx` et `data/benchmarks/MergedRawData.xlsx`. Parmi les études représentatives du benchmark :

| ID | Titre de l'Article | Revue / Année | Type d'Échantillon | Verdict Invasivité Réel |
|---|---|---|---|---|
| 012017-jfwm-007 | Identification of Southeastern Bat Species Using Noninvasive Genetic Sampling | *J. Fish & Wildlife Management* (2017) | Fèces / Guano & Tissus | Invasif (capture au filet et biopsies) |
| 1-s2.0-S0006320713002772 | Combining camera-trapping and noninvasive genetic data in spatial capture-recapture | *Biological Conservation* (2013) | Poils (hair snares) & Fèces | Non-invasif (pièges passifs) |
| 2015_HarmsEtAl | Predator identification from saliva traces | *Journal of Mammalogy* (2015) | Salive sur carcasses | Non-invasif (traces salivaire passives) |
| CONICET_Digital_Nro.14085 | Fecal DNA sampling of carnivores using detection dogs | *Mammalian Biology* (2014) | Fèces (chiens de détection) | Invasif - Péché 1 (stress/dérangement) |
| Marlowetal2015AJZ | Genetic monitoring of small dasyurids using hair tubes | *Australian Journal of Zoology* (2015) | Poils (tubes adhésifs) | Non-invasif |
