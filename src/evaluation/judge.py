"""Module d'évaluation automatisée LLM-as-a-Judge."""

import os
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

from src.rag.UniversityLLMAdapter import UniversityLLMAdapter
from src.common.config import get_api_key, get_api_url, get_model_name, BENCHMARKS_DIR
from src.evaluation.metrics import analyser_verdicts, generer_rapport_texte


def normaliser_texte(texte: Any) -> str:
    """Normalise un texte pour comparaison phonétique/lexicale."""
    if pd.isna(texte):
        return ""
    texte_str = str(texte)
    norm = unicodedata.normalize('NFKD', texte_str)
    norm = norm.encode('ascii', 'ignore').decode('utf-8')
    return norm.lower().strip()


class LLMJudge:
    """Juge automatique basé sur un LLM pour comparer les sorties du RAG à la vérité terrain."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key if api_key is not None else get_api_key()
        self.api_url = api_url if api_url is not None else get_api_url()
        self.model = model if model is not None else get_model_name()
        self.llm = UniversityLLMAdapter(
            api_key=self.api_key,
            api_url=self.api_url,
            model=self.model,
            max_tokens=1024,
            temperature=0.1
        )

    def align_predictions_with_ground_truth(
        self,
        df_pred: pd.DataFrame,
        df_ground_truth: pd.DataFrame,
        col_pred_title: str = "Titre_extrait_JSON",
        col_gt_title: str = "Title"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
        """Apparie les prédictions et la vérité terrain via score de similarité de titre."""
        df_pred_norm = df_pred.copy()
        df_pred_norm["title_norm"] = df_pred_norm[col_pred_title].apply(normaliser_texte)
        df_gt_norm = df_ground_truth.copy()
        df_gt_norm["title_norm"] = df_gt_norm[col_gt_title].apply(normaliser_texte)

        gt_titles = df_gt_norm["title_norm"].tolist()
        matches = []

        for i, title in enumerate(df_pred_norm["title_norm"]):
            if not title:
                matches.append({"pred_idx": i, "gt_idx": None, "score": 0.0})
                continue

            res = process.extractOne(title, gt_titles, scorer=fuzz.token_sort_ratio)
            if res:
                match_val, score, idx = res
                matches.append({"pred_idx": i, "gt_idx": idx, "score": score})
            else:
                matches.append({"pred_idx": i, "gt_idx": None, "score": 0.0})

        pred_matched = df_pred.copy().reset_index(drop=True)
        gt_matched = pd.DataFrame(
            [df_ground_truth.iloc[m["gt_idx"]] if m["gt_idx"] is not None else pd.Series(dtype='object') for m in matches]
        ).reset_index(drop=True)
        scores = [m["score"] for m in matches]

        return pred_matched, gt_matched, scores

    def evaluate_sample_matching(
        self,
        pred_matched: pd.DataFrame,
        gt_matched: pd.DataFrame,
        col_pred_sample: str = "Protocole",
        col_gt_sample: str = "What they called non-invaisve SAMPLES"
    ) -> List[str]:
        """Juge si l'échantillon biologique extrait correspond à la vérité terrain."""
        verdicts = []
        for idx in tqdm(range(len(pred_matched)), desc="Evaluation Echantillon"):
            row_pred = pred_matched.loc[idx]
            row_gt = gt_matched.loc[idx]

            prompt = f"""Tu es un assistant scientifique rigoureux en biologie moléculaire.
On te donne deux descriptions d'échantillons biologiques issus d'articles de recherche :
- Ligne expérimentale automatique : {row_pred.get(col_pred_sample, 'N/A')}
- Vérité terrain d'experts : {row_gt.get(col_gt_sample, 'N/A')}

Ta tâche est de juger si les deux descriptions parlent essentiellement du même type d'échantillon biologique.
Si l'information automatique est plus précise mais cohérente avec la vérité terrain, réponds OUI.
Réponds uniquement par 'Oui' ou 'Non', suivi d'une courte justification."""

            verdict = self.llm.generate_answer(prompt)
            verdicts.append(verdict.strip())
        return verdicts

    def evaluate_conclusion_matching(
        self,
        pred_matched: pd.DataFrame,
        gt_matched: pd.DataFrame,
        col_pred_eval: str = "Évaluation d'invasivité",
        col_gt_eval: str = "Compliance with Taberlet"
    ) -> List[str]:
        """Juge si la conclusion d'invasivité (Invasif / Non-invasif) est concordante."""
        verdicts = []
        for idx in tqdm(range(len(pred_matched)), desc="Evaluation Conclusion"):
            row_pred = pred_matched.loc[idx]
            row_gt = gt_matched.loc[idx]

            prompt = f"""Tu es un assistant scientifique expert en éthique et méthodes d'échantillonnage de la faune sauvage (Taberlet et al. 1999).
Compare les deux conclusions d'invasivité suivantes :
- Évaluation automatique : {row_pred.get(col_pred_eval, 'N/A')}
- Vérité terrain annotée : {row_gt.get(col_gt_eval, 'N/A')}

La conclusion d'invasivité (invasif vs non-invasif) est-elle identique entre les deux ?
Réponds uniquement par 'Oui' ou 'Non'."""

            verdict = self.llm.generate_answer(prompt)
            verdicts.append(verdict.strip())
        return verdicts
