"""Module de calcul des métriques d'évaluation et génération de rapports statistiques."""

import pandas as pd
from typing import Dict, Any, List


def analyser_verdicts(df: pd.DataFrame, colonne_verdict: str, nom_verdict: str) -> Dict[str, Any]:
    """
    Analyse les verdicts LLM-as-a-Judge pour un type donné (Échantillon, Protocole ou Conclusion).
    Calcule les métriques de Précision, Rappel et F1-Score au niveau ligne et au niveau article.
    """
    df_copy = df.copy()
    df_copy[colonne_verdict] = df_copy[colonne_verdict].astype(str).str.strip().str.lower()
    df_copy['verdict_binaire'] = df_copy[colonne_verdict].apply(lambda x: 1 if 'oui' in x else 0)

    total_lignes = len(df_copy)
    nb_oui_lignes = int((df_copy['verdict_binaire'] == 1).sum())
    nb_non_lignes = int((df_copy['verdict_binaire'] == 0).sum())

    # Regroupement par article
    col_titre = 'Titre_verite' if 'Titre_verite' in df_copy.columns else df_copy.columns[0]
    stats_par_article = df_copy.groupby('ID_article').agg({
        'verdict_binaire': ['count', 'sum'],
        col_titre: 'first'
    }).reset_index()

    stats_par_article.columns = ['ID_article', 'nb_lignes', 'nb_oui', 'Titre_verite']
    stats_par_article['nb_non'] = stats_par_article['nb_lignes'] - stats_par_article['nb_oui']
    stats_par_article['verdict_global'] = stats_par_article['nb_oui'].apply(
        lambda x: 'Oui' if x > 0 else 'Non'
    )

    total_articles = len(stats_par_article)
    articles_oui = int((stats_par_article['verdict_global'] == 'Oui').sum())
    articles_non = int((stats_par_article['verdict_global'] == 'Non').sum())

    # Métriques niveau ligne
    tp_lignes = nb_oui_lignes
    fn_lignes = nb_non_lignes
    total_lignes_attendues = tp_lignes + fn_lignes
    precision_lignes = 100.0 if tp_lignes > 0 else 0.0
    rappel_lignes = (tp_lignes / total_lignes_attendues * 100.0) if total_lignes_attendues > 0 else 0.0
    f1_lignes = (2 * precision_lignes * rappel_lignes / (precision_lignes + rappel_lignes)) if (precision_lignes + rappel_lignes) > 0 else 0.0

    # Métriques niveau article
    tp_articles = articles_oui
    fn_articles = articles_non
    total_articles_attendus = tp_articles + fn_articles
    precision_articles = 100.0 if tp_articles > 0 else 0.0
    rappel_articles = (tp_articles / total_articles_attendus * 100.0) if total_articles_attendus > 0 else 0.0
    f1_articles = (2 * precision_articles * rappel_articles / (precision_articles + rappel_articles)) if (precision_articles + rappel_articles) > 0 else 0.0

    return {
        "nom": nom_verdict,
        "colonne": colonne_verdict,
        "total_lignes": total_lignes,
        "tp_lignes": tp_lignes,
        "fn_lignes": fn_lignes,
        "precision_lignes": precision_lignes,
        "rappel_lignes": rappel_lignes,
        "f1_lignes": f1_lignes,
        "total_articles": total_articles,
        "tp_articles": tp_articles,
        "fn_articles": fn_articles,
        "precision_articles": precision_articles,
        "rappel_articles": rappel_articles,
        "f1_articles": f1_articles,
        "details_articles": stats_par_article
    }


def generer_rapport_texte(stats: Dict[str, Any]) -> str:
    """Génère un rapport texte rigoureux et sans émojis."""
    lignes: List[str] = [
        "=" * 70,
        f"STATISTIQUES - {stats['nom'].upper()}",
        "=" * 70,
        "",
        "[RESUME GLOBAL]",
        f"- Nombre total de lignes : {stats['total_lignes']}",
        f"- Lignes avec verdict 'Oui' (TP) : {stats['tp_lignes']} ({stats['tp_lignes'] / stats['total_lignes'] * 100:.1f}%)",
        f"- Lignes avec verdict 'Non' (FN) : {stats['fn_lignes']} ({stats['fn_lignes'] / stats['total_lignes'] * 100:.1f}%)",
        f"- Nombre d'articles differents : {stats['total_articles']}",
        "",
        "[SYNTHESE FINALE]",
        f"- Articles avec verdict 'Oui' : {stats['tp_articles']}/{stats['total_articles']} ({stats['rappel_articles']:.1f}%)",
        f"- Articles avec verdict 'Non' : {stats['fn_articles']}/{stats['total_articles']} ({100.0 - stats['rappel_articles']:.1f}%)",
        "",
        "[METRIQUES DE PERFORMANCE]",
        "",
        "Niveau Ligne (granularite fine) :",
        f"  - True Positives (TP)  : {stats['tp_lignes']} lignes extraites",
        f"  - False Negatives (FN) : {stats['fn_lignes']} lignes ratees",
        f"  - Precision            : {stats['precision_lignes']:.2f}%",
        f"  - Rappel               : {stats['rappel_lignes']:.2f}%",
        f"  - F1-Score             : {stats['f1_lignes']:.2f}%",
        "",
        "Niveau Article (verdict global) :",
        f"  - Articles reussis (TP): {stats['tp_articles']}",
        f"  - Articles rates (FN)  : {stats['fn_articles']}",
        f"  - Precision            : {stats['precision_articles']:.2f}%",
        f"  - Rappel               : {stats['rappel_articles']:.2f}%",
        f"  - F1-Score             : {stats['f1_articles']:.2f}%",
        "=" * 70
    ]
    return "\n".join(lignes)
