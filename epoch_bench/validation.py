"""Expert review export/import and inter-annotator agreement."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from epoch_bench.schema import Question


def export_for_review(
    questions: list[Question],
    path: str | Path,
    format: str = "csv",
) -> None:
    """Export questions for expert review with empty reviewer columns."""
    path = Path(path)

    rows = []
    for q in questions:
        rows.append(
            {
                "id": q.id,
                "type": q.type.value,
                "variant": q.variant,
                "pair_id": q.pair_id,
                "prompt": q.prompt,
                "answer": json.dumps(q.answer) if isinstance(q.answer, list) else q.answer,
                "difficulty": q.difficulty or "",
                "domains": ", ".join(q.domains) if q.domains else "",
                "reviewer_score": "",
                "reviewer_notes": "",
                "validated": "",
            }
        )

    if format == "csv":
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    elif format == "json":
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def import_reviews(path: str | Path) -> list[dict]:
    """Parse completed reviews back from CSV or JSON."""
    path = Path(path)

    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)

    # CSV
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        reviews = []
        for row in reader:
            score = row.get("reviewer_score", "")
            if score and score.strip():
                row["reviewer_score"] = int(score.strip())
            else:
                row["reviewer_score"] = None
            reviews.append(row)
        return reviews


def inter_annotator_agreement(reviews: list[list[dict]]) -> dict:
    """Cohen's kappa (2 annotators) or Fleiss' kappa (3+)."""
    if len(reviews) < 2:
        return {"kappa": 1.0, "type": "none", "n_items": 0}

    id_sets = [set(r["id"] for r in ann) for ann in reviews]
    common_ids = sorted(set.intersection(*id_sets))

    if not common_ids:
        return {"kappa": 0.0, "type": "none", "n_items": 0}

    annotator_scores: list[dict[str, int | None]] = []
    for ann in reviews:
        score_map = {}
        for r in ann:
            score_map[r["id"]] = r.get("reviewer_score")
        annotator_scores.append(score_map)

    if len(reviews) == 2:
        return _cohens_kappa(annotator_scores[0], annotator_scores[1], common_ids)
    return _fleiss_kappa(annotator_scores, common_ids)


def _cohens_kappa(
    scores_a: dict[str, int | None],
    scores_b: dict[str, int | None],
    common_ids: list[str],
) -> dict:
    """Cohen's kappa for two annotators."""
    pairs = []
    for qid in common_ids:
        sa, sb = scores_a.get(qid), scores_b.get(qid)
        if sa is not None and sb is not None:
            pairs.append((sa, sb))

    if not pairs:
        return {"kappa": 0.0, "type": "cohen", "n_items": 0}

    n = len(pairs)
    categories = sorted(set(a for a, _ in pairs) | set(b for _, b in pairs))

    # Observed agreement
    po = sum(1 for a, b in pairs if a == b) / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        p_a = sum(1 for a, _ in pairs if a == cat) / n
        p_b = sum(1 for _, b in pairs if b == cat) / n
        pe += p_a * p_b

    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {"kappa": kappa, "type": "cohen", "n_items": n}


def _fleiss_kappa(
    annotator_scores: list[dict[str, int | None]],
    common_ids: list[str],
) -> dict:
    """Fleiss' kappa for 3+ annotators."""
    k = len(annotator_scores)

    valid_items = []
    for qid in common_ids:
        scores = [ann.get(qid) for ann in annotator_scores]
        if all(s is not None for s in scores):
            valid_items.append((qid, scores))

    if not valid_items:
        return {"kappa": 0.0, "type": "fleiss", "n_items": 0}

    n = len(valid_items)
    all_scores = [s for _, scores in valid_items for s in scores]
    categories = sorted(set(all_scores))
    cat_idx = {c: i for i, c in enumerate(categories)}
    c = len(categories)

    if c <= 1:
        return {"kappa": 1.0, "type": "fleiss", "n_items": n}

    count_matrix = []
    for _, scores in valid_items:
        row = [0] * c
        for s in scores:
            row[cat_idx[s]] += 1
        count_matrix.append(row)

    p_items = []
    for row in count_matrix:
        sum_sq = sum(x * x for x in row)
        p_i = (sum_sq - k) / (k * (k - 1))
        p_items.append(p_i)

    p_bar = sum(p_items) / n

    total_assignments = n * k
    p_cats = []
    for j in range(c):
        col_sum = sum(count_matrix[i][j] for i in range(n))
        p_cats.append(col_sum / total_assignments)

    pe = sum(p**2 for p in p_cats)

    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (p_bar - pe) / (1.0 - pe)

    return {"kappa": kappa, "type": "fleiss", "n_items": n}


def filter_validated(
    questions: list[Question],
    reviews: list[dict],
    min_score: int = 3,
) -> list[Question]:
    """Keep only questions with reviewer_score >= min_score."""
    validated_ids = set()
    for r in reviews:
        score = r.get("reviewer_score")
        if score is not None and score >= min_score:
            validated_ids.add(r["id"])

    return [q for q in questions if q.id in validated_ids]
