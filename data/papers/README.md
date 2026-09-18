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

---

## 4. Bibliographie Scientifique Complète (Issu de `Stage L3 RAG.xml`)

L'ensemble des publications ci-dessous est extrait directement de la bibliothèque Zotero de référence du projet (`Stage L3 RAG.xml`) :

### Échantillonnage Non-Invasif et Éthique de la Faune
1. **Lefort, M.-C., Cruickshank, R. H., Descovich, K., et al.** (2022). *Blood, sweat and tears: a review of non-invasive DNA sampling*. Peer Community Journal, 2, e98. [DOI: 10.24072/pcjournal.98](https://doi.org/10.24072/pcjournal.98).
2. **Taberlet, P., Waits, L. P., & Luikart, G.** (1999). *Noninvasive genetic sampling: look before you leap*. Trends in Ecology & Evolution, 14(8), 323–327. [DOI: 10.1016/S0169-5347(99)01637-7](https://doi.org/10.1016/S0169-5347(99)01637-7).
3. **Calvignac-Spencer, S., Merkel, K., Kutzner, N., et al.** (2013). *Carrion fly-derived DNA as a tool for comprehensive and cost-effective assessment of mammalian biodiversity*. Molecular Ecology, 22(4), 915–924. [DOI: 10.1111/mec.12183](https://doi.org/10.1111/mec.12183).

### LLM-as-a-Judge et Évaluation Automatisée
4. **Gu, J., Jiang, X., Shi, Z., et al.** (2025). *A Survey on LLM-as-a-Judge*. arXiv:2411.15594. [arXiv:2411.15594](https://arxiv.org/abs/2411.15594).
5. **Shi, L., Ma, C., Liang, W., et al.** (2025). *Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge*. arXiv:2406.07791. [arXiv:2406.07791](https://arxiv.org/abs/2406.07791).
6. **Honovich, O., Choshen, L., Aharoni, R., et al.** (2021). *$Q^2$: Evaluating Factual Consistency in Knowledge-Grounded Dialogues via Question Generation and Question Answering*. arXiv:2104.08202. [arXiv:2104.08202](https://arxiv.org/abs/2104.08202).

### Architectures RAG, Chunking et Recherche Dense
7. **Gao, Y., Xiong, Y., Gao, X., et al.** (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997. [arXiv:2312.10997](https://arxiv.org/abs/2312.10997).
8. **Gao, L., Ma, X., Lin, J., & Callan, J.** (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*. arXiv:2212.10496. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496).
9. **Gao, Y., Xiong, Y., Wu, W., et al.** (2025). *U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-In-A-Haystack*. arXiv:2503.00353. [arXiv:2503.00353](https://arxiv.org/abs/2503.00353).
10. **Jin, B., Yoon, J., Han, J., et al.** (2024). *Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG*. arXiv:2410.05983. [arXiv:2410.05983](https://arxiv.org/abs/2410.05983).
11. **Günther, M., Mohr, I., Williams, D. J., et al.** (2024). *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*. arXiv:2409.04701. [arXiv:2409.04701](https://arxiv.org/abs/2409.04701).
12. **Wang, L., Chen, H., Yang, N., et al.** (2025). *Chain-of-Retrieval Augmented Generation*. arXiv:2501.14342. [arXiv:2501.14342](https://arxiv.org/abs/2501.14342).
13. **Zhao, W. X., Liu, J., Ren, R., et al.** (2022). *Dense Text Retrieval based on Pretrained Language Models: A Survey*. arXiv:2211.14876. [arXiv:2211.14876](https://arxiv.org/abs/2211.14876).
14. **Zhao, P., Zhang, H., Yu, Q., et al.** (2024). *Retrieval-Augmented Generation for AI-Generated Content: A Survey*. arXiv:2402.19473. [arXiv:2402.19473](https://arxiv.org/abs/2402.19473).
15. **Xu, P., Ping, W., Wu, X., et al.** (2024). *Retrieval Meets Long Context Large Language Models*.
16. **Merth, T., Fu, Q., Rastegari, M., et al.** (2024). *Superposition Prompting: Improving and Accelerating Retrieval-Augmented Generation*. ICML 2024. [OpenReview](https://openreview.net/forum?id=r8k5JrGip6).
17. **Wang, Y., Hou, Y., Wang, H., et al.** (2023). *A Neural Corpus Indexer for Document Retrieval*. arXiv:2206.02743. [arXiv:2206.02743](https://arxiv.org/abs/2206.02743).
18. **He, J., Liu, G., Zhu, B., et al.** (2025). *Context-Guided Dynamic Retrieval for Improving Generation Quality in RAG Models*. arXiv:2504.19436. [arXiv:2504.19436](https://arxiv.org/abs/2504.19436).
19. **He, X., Tian, Y., Sun, Y., et al.** *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering*.
20. **Haystack Documentation** (2025). *Hypothetical Document Embeddings (HyDE)*. [deepset.ai](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde).

### Extraction d'Information Scientifique, Agents et Ingénierie de Prompts
21. **Foppiano, L., Lambard, G., Amagasa, T., et al.** (2024). *Mining experimental data from materials science literature with large language models: an evaluation study*. Science and Technology of Advanced Materials: Methods, 4(1). [DOI: 10.1080/27660400.2024.2356506](https://doi.org/10.1080/27660400.2024.2356506).
22. **Neves, M., Klippert, A., Knöspel, F., et al.** (2023). *Automatic classification of experimental models in biomedical literature to support searching for alternative methods to animal experiments*. Journal of Biomedical Semantics, 14(1). [DOI: 10.1186/s13326-023-00292-w](https://doi.org/10.1186/s13326-023-00292-w).
23. **Lála, J., O'Donoghue, O., Shtedritski, A., et al.** (2023). *PaperQA: Retrieval-Augmented Generative Agent for Scientific Research*. arXiv:2312.07559. [arXiv:2312.07559](https://arxiv.org/abs/2312.07559).
24. **Skarlinski, M. D., Cox, S., Laurent, J. M., et al.** (2024). *Language agents achieve superhuman synthesis of scientific knowledge*. arXiv:2409.13740. [arXiv:2409.13740](https://arxiv.org/abs/2409.13740).
25. **Kim, S.** (2025). *MedBioLM: Optimizing Medical and Biological QA with Fine-Tuned Large Language Models and Retrieval-Augmented Generation*. arXiv:2502.03004. [arXiv:2502.03004](https://arxiv.org/abs/2502.03004).
26. **Kim, S., & Mazumder, R.** *Enhancing Scientific Reproducibility Through Automated BioCompute Object Creation Using Retrieval-Augmented Generation from Publications*.
27. **Yao, S., Zhao, J., Yu, D., et al.** (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).
28. **Wang, X., Wei, J., Schuurmans, D., et al.** (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171).
29. **Zhang, Z., Zhang, A., Li, M., et al.** (2022). *Automatic Chain of Thought Prompting in Large Language Models*. arXiv:2210.03493. [arXiv:2210.03493](https://arxiv.org/abs/2210.03493).
30. **Topsakal, O., & Akinci, T. C.** (2023). *Creating Large Language Model Applications Utilizing LangChain: A Primer on Developing LLM Apps Fast*. ICAENS 2023. [DOI: 10.59287/icaens.1127](https://doi.org/10.59287/icaens.1127).
31. **McCune, B. P., Tong, R. M., Dean, J. S., et al.** (1985). *RUBRIC: A System for Rule-Based Information Retrieval*. IEEE Transactions on Software Engineering, SE-11(9), 939–945. [DOI: 10.1109/TSE.1985.232827](https://doi.org/10.1109/TSE.1985.232827).
32. **Niu, C., Guan, Y., Wu, Y., et al.** (2024). *VeraCT Scan: Retrieval-Augmented Fake News Detection with Justifiable Reasoning*. arXiv:2406.10289. [arXiv:2406.10289](https://arxiv.org/abs/2406.10289).
33. **Nurzhanov, A.** (2025). *Application of the Rag Concept for Detecting and Classifying Extremist Content in the Kazakh Language*. SSRN. [DOI: 10.2139/ssrn.5084790](https://doi.org/10.2139/ssrn.5084790).

