import pandas as pd
import os
import functools
from rapidfuzz import process, fuzz


_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@functools.lru_cache(maxsize=1)
def load_base_code_postal() -> pd.DataFrame:
    """
    Charge le CSV une seule fois en mémoire (mis en cache).
    """
    path = os.path.join(_DATA_DIR, "regions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path, dtype=str)
    df["nom_commune_upper"] = df["nom_commune"].str.strip().str.upper()
    return df


def code_postal_to_communes(code_postal: str) -> list[str]:
    """
    Retourne la liste des communes associées à un code postal.
    Retourne une liste vide si aucun résultat.

    Exemple :
        >>> code_postal_to_communes("75001")
        ["Paris"]
    """
    df = load_base_code_postal()
    results = df[df["code_postal"] == code_postal.strip()]["nom_commune"].tolist()
    return results


def commune_to_code_postal(nom_commune: str, threshold: int = 80) -> dict:
    """
    Convertit un nom de commune (approché) en code postal via matching flou.
    Retourne toujours un dict :
      - En succès : {"nom_commune": ..., "code_postal": ..., "score": ...}
      - En échec  : {"error": "...message explicite..."}

    Paramètres :
        nom_commune (str) : nom de la commune saisi par l'utilisateur
        threshold   (int) : score minimum de similarité (0-100), défaut 80

    Exemple :
        >>> commune_to_code_postal("Marseil")
        {"nom_commune": "Marseille", "code_postal": "13000", "score": 93}

        >>> commune_to_code_postal("xyzabc")
        {"error": "Commune introuvable : 'xyzabc'. Vérifiez l'orthographe ou essayez un nom approché."}
    """
    df = load_base_code_postal()
    communes_upper = df["nom_commune_upper"].dropna().tolist()

    result = process.extractOne(
        nom_commune.strip().upper(),
        communes_upper,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold
    )

    if result is None:
        return {
            "error": f"Commune introuvable : '{nom_commune}'. Vérifiez l'orthographe ou essayez un nom approché."
        }

    best_match, score, idx = result
    row = df.iloc[idx]

    return {
        "nom_commune": row["nom_commune"],
        "code_postal": row["code_postal"],
        "score": score
    }