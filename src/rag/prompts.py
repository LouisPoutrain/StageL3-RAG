"""
Ce fichier centralise tous les prompts utilisés par le système RAG.

"""

# ==============================================================================
# Prompts pour RefineRAGSystem
# ==============================================================================

PROTOCOL_DETECTION_PROMPT = """
Tu es un expert en protocoles d'échantillonnage d'ADN.

Ta mission est d'analyser l'extrait ci-dessous et de :

1. Identifier s'il contient une ou plusieurs descriptions de **collecte ou de traitement** d'échantillons biologiques (sang, poils, fèces, urine, tissus, etc.).
- Une description peut être explicite ("des poils ont été prélevés") ou implicite ("les otaries ont été capturées et échantillonnées").
- Les informations sur la manipulation, le stockage, la subsampling, ou l'extraction sont également valides.
- La collecte de fèces, même sans capture directe, est considérée comme un protocole d'échantillonnage.

2. Si un ou plusieurs protocoles sont détectés, renvoyer **uniquement** ce JSON :
```json{{
"contient_protocole": true,
"extrait_pertinent": [
    "Citation exacte 1 du protocole",
    "Citation exacte 2 (s'il y en a un autre)",
    ...
]
}}```

Si aucun protocole n'est détecté dans l'extrait, renvoyer uniquement :

{{
"contient_protocole": false
}}

Tu dois raisonner en interne comme un expert : repérer les verbes ou actions typiques, réfléchir à l'objectif de chaque phrase, mais ne jamais afficher tes pensées. Seule la réponse JSON doit apparaître.
Attention à bien extraire toutes les citations pertinentes, même si elles sont dans des phrases différentes. Attention à bien extraire les citations qui décrivent le protocole, pas leur conservation...
De plus, quand tu extrais les citations, tu dois les extraire dans leur intégralité ou suffisamment pour comprendre le protocole.

Extrait à analyser :
{chunk}

"""

FUSION_EXTRAIT_PROMPT = """
Tu es un expert en analyse de protocoles scientifiques dans le domaine de l'échantillonnage d'ADN.
On te fournit un ensemble d'extraits pertinents provenant d'un même document scientifique.
Ta tâche est de regrouper les extraits qui décrivent le même protocole d'échantillonnage ADN.

Règles de fusion des extraits :

1. Critères de fusion simple :
   - Regrouper tous les extraits qui parlent du même type de prélèvement
   - Exemples :
     * Si un extrait parle de prélèvement de fèces, regrouper avec tous les autres extraits qui parlent de fèces
     * Si un extrait parle de prélèvement d'eau, regrouper avec tous les autres extraits qui parlent d'eau
     * Si un extrait parle de prélèvement de poils, regrouper avec tous les autres extraits qui parlent de poils

2. Gestion des extraits :
   - Conserver uniquement les extraits les plus détaillés et informatifs
   - Éliminer les extraits redondants ou moins informatifs
   - Si deux extraits sont similaires mais l'un est plus détaillé, garder le plus détaillé
   - Si deux extraits sont similaires mais décrivent des variantes importantes, garder les deux

{{
"contient_protocole": true,
"extrait_pertinent": [
    "Citation exacte 1 du protocole",
    "Citation exacte 2",
    ...
]
}}

3. Règles anti-hallucination :
   - Ne pas inventer d'informations
   - Ne pas modifier le sens des extraits
   - Ne pas fusionner des extraits qui décrivent des protocoles différents
   - Ne pas ajouter de détails non présents dans les extraits

Extrait à analyser :
{extracted_chunks}

Tu dois uniquement renvoyer le JSON, aucune autre explication, aucun commentaire, aucune phrase.
Un JSON par protocole fusionné.
"""

SEVEN_SINS_DEFINITIONS = """
Voici les définitions des Péchés de l'Échantillonnage d'ADN Non-Invasif :
1. Mauvaise classification des fèces comme échantillons d'ADN non invasifs : Considérer automatiquement la collecte de fèces comme non invasive sans tenir compte du contexte de la collecte (par exemple, capture de l'animal pour obtenir les fèces, utilisation d'aéronefs pouvant stresser les animaux, impact sur le marquage territorial).
2. Appâtage des pièges à ADN : Utiliser des appâts ou des leurres pour augmenter le rendement des pièges à ADN, ce qui modifie le comportement naturel des animaux et ne correspond pas à une méthode entièrement non invasive.
3. Un oiseau dans la main vaut mieux que deux dans la nature (Capture ou manipulation d'animaux) : Capturer et/ou manipuler des animaux sauvages pour obtenir des échantillons d'ADN, ce qui cause du stress et potentiellement des blessures, contrairement à la définition de l'échantillonnage non invasif.
4. Tout ou rien (Manque de reconnaissance des approches minimalement invasives) : Ne pas reconnaître ou utiliser le terme "minimalement invasif" pour des méthodes qui réduisent l'impact sur l'animal, car la définition stricte de "non invasif" ne laisse pas de place au milieu.
5. Équivalence entre une procédure non invasive et un échantillonnage d'ADN non invasif : Utiliser la définition médicale ou vétérinaire d'une procédure non invasive (qui n'implique pas de perforation de la peau) pour classer l'échantillonnage d'ADN, sans tenir compte de l'impact comportemental ou du bien-être animal.
"""

INVASIVITY_ANALYSIS_PROMPT = """
Tu es un expert en analyse de protocoles d'échantillonnage d'ADN.

Ta tâche est d'analyser le protocole suivant pour déterminer s'il est invasif selon la définition donnée.

Définition de l'échantillonnage non-invasif :
{definition}

Définition des péchés de l'échantillonnage d'ADN :
""" + SEVEN_SINS_DEFINITIONS + """

Critères d'invasivité :
Un protocole est invasif si au moins UN des critères suivants est rempli:
- Contact direct avec l'animal vivant
- Capture ou manipulation de l'animal vivant
- Modification du comportement naturel
- Perturbation significative de l'habitat
- Utilisation d'appâts ou leurres modifiant le comportement


Un protocole est non-invasif UNIQUEMENT si TOUS ces critères sont remplis:
- Aucun contact avec l'animal vivant
- Aucune perturbation du comportement naturel
- Utilisation d'échantillons déjà abandonnés naturellement
- Aucune modification de l'habitat ou du territoire

Cas particuliers de non invasivité:
- Manipulation d'animaux morts naturellement.
- Si la collecte est faite après le départ de l'animal sans capture ou manipulation.
- Si des données existantes sont ajoutées à des échantillons déjà existants et que celle si sont invasifs, et que l'ajout n'est pas invasif, alors le protocole est non invasif.

Règles spécifiques par type d'échantillon et espèce
Fèces de mammifères :
Non invasif : Collecte ponctuelle, opportuniste, partielle (< 50% des fèces observées)
Invasif - Territory marking : Collecte systématique, exhaustive, régulière sur même territoire
Indicateurs d'invasivité : "toutes les fèces", "collecte hebdomadaire/quotidienne", "sur l'ensemble du transect", "nettoyage complet"

Autres échantillons par taxon :
Oiseaux : Plumes naturellement perdues (non invasif) vs plumes arrachées (invasif)
Reptiles/Amphibiens : Mues naturelles (non invasif) vs manipulation pour prélèvement (invasif)
Poissons : Écailles perdues naturellement (non invasif) vs capture pour prélèvement (invasif)
Invertébrés : Exuvies naturelles (non invasif) vs capture (invasif)

ADN environnemental (eDNA) :
Non invasif : Prélèvement d'eau, sédiments, sol sans perturbation
Invasif : Prélèvement nécessitant perturbation d'habitat, drainage, excavation

Processus d'analyse avec ReAct :

        1. Réflexion (Thought) :
        Identifier le taxon étudié (mammifère, oiseau, etc.)
        Identifier le type d'échantillon (fèces, plumes, ADNe, etc.)
        Rechercher les mots-clés d'invasivité : "capture", "manipulation", "toutes", "systématique", "exhaustif", ...
        Rechercher les mots-clés de non-invasivité : "naturellement", "opportuniste", "ponctuel", "abandonné", ...
        Évaluer la fréquence et l'intensité de l'échantillonnage

        Attention : des termes comme "systématique", "quotidien", "exhaustif" n'indiquent une invasivité que s'ils concernent un type d'échantillon pouvant affecter le comportement ou le territoire. Par exemple, une collecte systématique de plumes abandonnées peut rester non invasive si elle n'entraîne ni perturbation ni manipulation.

        2. Action (Act) :
        Appliquer les règles spécifiques au taxon et type d'échantillon
        Évaluer chaque critère d'invasivité
        Déterminer le niveau de certitude basé sur :

        Explicite (90-100%) : Méthodologie détaillée, termes clairs
        Implicite fort (70-89%) : Éléments permettant déduction solide
        Implicite faible (50-69%) : Éléments partiels, interprétation nécessaire
        Ambigu (30-49%) : Informations contradictoires ou insuffisantes
        Absence (0-29%) : Pas d'information sur la méthodologie


        3. Observation (Obs) :
        Citer les extraits avec contexte suffisant (phrase complète + contexte)
        Identifier les signaux d'alarme ou de validation
        Noter les informations manquantes critiques

    Signaux d'alarme pour l'invasivité (à contextualiser):

    Fréquence : "quotidien", "hebdomadaire", "systématique", "régulier"
    Exhaustivité : "toutes", "ensemble", "complet", "total"
    Manipulation : "capture", "manipulation", "contention", "anesthésie"
    Perturbation : "piégeage", "appât", "leurre", "dérangement"

    La fréquence et l'exhaustivité concerne uniquement le prélèvement de fèces chez les mamifères

    Signaux de non-invasivité :

    "Naturellement perdues/abandonnées"
    "Opportuniste", "ponctuel"
    "Après départ de l'animal"
    "Échantillons trouvés"

Cas ambigus - Règles de décision :

Information manquante critique → Taux de confiance < 50%
Contradiction dans le texte → Retenir l'interprétation la plus conservative (invasif)
Protocole mixte → Classifier selon l'élément le plus invasif
Espèce non-mammifère + fèces → Appliquer les règles générales (pas de règle territoriale spécifique)



Protocole à analyser :
{protocol}

Renvoie ton analyse au format JSON suivant :
```json
{{
"extrait_pertinent": ["Citation exacte avec contexte suffisant 1", "Citation exacte avec contexte suffisant 2"],
"taxon_identifie": "Mammifère/Oiseau/Reptile/Poisson/Invertébré/Mixte/Non précisé",
"echantillon": "Type d'échantillon précis",
"mots_cles_invasivite": ["mot-clé 1", "mot-clé 2"],
"mots_cles_non_invasivite": ["mot-clé 1", "mot-clé 2"],
"evaluation_invasivite": "Invasif", "Non invasif" ou "Invasif - Territory marking",
"justification_invasivite": "Justification détaillée basée sur les critères et règles spécifiques au taxon",
"signaux_alarme": ["Signal d'alarme 1", "Signal d'alarme 2"],
"impacts_potentiels": ["Impact 1", "Impact 2"],
"peches_identifies": ["1", "2", "5"],
"informations_manquantes": ["Information critique manquante 1"],
"taux_de_confiance": "Taux entre 0 et 100 avec justification du niveau"
}}
```

Tu dois Uniquement renvoyer le JSON, aucune autre explication.
Tu dois analyser tous les protocoles fournis.
Si aucun protocole n'est fourni, ne renvoie rien.
"""

FUSION_PROMPT = """
Tu es un expert en protocoles d'échantillonnage d'ADN.

Ta tâche est de fusionner les analyses suivantes en un tableau JSON final.

Règles de fusion :
1. Si deux protocoles décrivent la même procédure (c'est-à-dire même méthode de collecte, même espèce, même type d'échantillon, mêmes conditions de terrain), même si le texte varie légèrement, les fusionner.
2. Conserver tous les extraits pertinents.
3. Uniformiser les noms d'échantillons similaires.
4. Ne pas inventer d'informations.
5. Retire les protocoles inconnus avec aucune citation.
6. Conserver l'évaluation d'invasivité la plus stricte (Invasif > Non invasif).


Processus de fusion avec ReAct :

1. Filtrage (Filtering) :
- Garde uniquement les protocoles liés à l'échantillonnage d'ADN.
- Ignore tout ce qui ne décrit pas explicitement une procédure d'échantillonnage.
- Un protocole doit mentionner la collecte ou l'analyse de matériel biologique contenant de l'ADN.

2. Fusion (Fusion) :
- Si deux protocoles décrivent la même procédure, les fusionner.
- Conserver tous les extraits pertinents.
- Uniformiser les noms d'échantillons similaires.
- Ne pas inventer d'informations.

3. Évaluation (Evaluation) :
- Uniformise les noms d'échantillons très proches.
- Résoudre les conflits en fonction des règles strictes.

4. Résolution de conflits (Conflict Resolution) :
- Si un protocole est appliqué sur des animaux déjà morts de cause naturelle, les impacts doivent être considérés comme nuls.
- L'évaluation de l'invasivité doit être "Non invasif".
- Les péchés doivent être une liste vide.
- Conserver l'évaluation d'invasivité la plus stricte (Invasif > Non invasif).
- Cette règle est impérative sauf si un extrait directement contradictoire et explicite permet d'inférer une réduction claire de l'impact.
- Si l'évaluation d'invasivité est "Invasif" à cause d'une collecte supposée systématique, mais que d'autres extraits indiquent une collecte partielle, non systématique ou respectueuse du marquage, alors l'évaluation peut être corrigée en "Non invasif", et le péché retiré.

5. Format de sortie (Output Format) :
- Génère un seul tableau JSON, strictement conforme au format suivant.
- Ne produis aucun texte avant ou après le JSON.

6. Le champ protocole doit indiquer la méthode ou l'espèce ciblée.

Analyses à fusionner :
{analyses}

Renvoie le résultat au format JSON suivant :
```json
[
    {{
        "protocole": "Nom précis du protocole",
        "extrait_pertinent": ["Citation exacte avec contexte suffisant 1", "Citation exacte avec contexte suffisant 2"],
        "taxon_identifie": "Mammifère/Oiseau/Reptile/Poisson/Invertébré/Mixte/Non précisé",
        "echantillon": "Type d'échantillon précis",
        "mots_cles_invasivite": ["mot-clé 1", "mot-clé 2"],
        "mots_cles_non_invasivite": ["mot-clé 1", "mot-clé 2"],
        "evaluation_invasivite": "Invasif" ou "Non invasif",
        "justification_invasivite": "Justification détaillée basée sur les critères et règles spécifiques au taxon",
        "signaux_alarme": ["Signal d'alarme 1", "Signal d'alarme 2"],
        "impacts_potentiels": ["Impact 1", "Impact 2"],
        "peches_identifies": ["1", "2", "5"],
        "informations_manquantes": ["Information critique manquante 1"],
        "taux_de_confiance": "Taux entre 0 et 100 avec justification du niveau"
    }}
]
```

Tu dois Uniquement renvoyer le JSON, aucune autre explication, aucun commentaire, aucune phrase.
Attention, l'apostrophe ne s'échappe pas en JSON, pas besoin de mettre de backslash devant.
"""


# ==============================================================================
# Prompts pour RAGNonInvasiveDetection
# ==============================================================================

NON_INVASIVE_DETECTION_PROMPT = """
Voici un extrait d'article scientifique :

\"\"\"{Chunks}\"\"\"

Le protocole suivant a été détecté comme étant invasif par une analyse externe :
{protocole}

Ta tâche est de déterminer si l'auteur de l'article affirme que ce protocole est non invasif, de manière explicite ou implicite, dans ce passage.

Voici le titre de l'article :
"{title}"

Procède étape par étape :

1. Analyse le titre : contient-il une déclaration globale de non-invasivité ? Si oui, réponds "Oui" d'office.
2. Analyse si le titre mentionne le protocole détecté comme invasif. Si ce protocole est présenté dans le titre comme non invasif, réponds aussi "Oui" d'office.
3. Cherche dans le corps de l'article :
- Y a-t-il une phrase indiquant explicitement que le protocole en question est non invasif ? Si oui, réponds "Oui".
- Si une autre méthode est dite non invasive, mais ce n'est **pas celle détectée comme invasive, ce n'est pas suffisant.
4. Si aucune déclaration claire ou implicite n'est trouvée, réponds "Non".
5. Si le passage est trop vague ou ne permet pas de trancher, réponds "Manque d'informations".

Tu dois répondre uniquement par le JSON suivant :
```json
{{
    "annonce_invasivite": "Oui" ou "Non" ou "Manque d'informations",
    "justification_annonce_invasivite": "Justification courte, avec citation si utile (souvent utile pour la justification)"
}}
```
Si tu ne peux pas fournir une réponse, réponds "Pas de réponse".

Ne donne aucune explication hors du JSON, aucune phrase d'introduction ou de conclusion.
les clés doivent être strictement : annonce_invasivite et justification_annonce_invasivite
""" 
