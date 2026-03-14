"""Tests for epoch_bench.validation review export/import and agreement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epoch_bench.schema import Question, QuestionType
from epoch_bench.validation import (
    export_for_review,
    filter_validated,
    import_reviews,
    inter_annotator_agreement,
)


@pytest.fixture
def review_questions() -> list[Question]:
    return [
        Question(
            id="q1",
            type=QuestionType.GATE,
            variant="factual",
            pair_id="p1",
            prompt="Test?",
            answer="Yes",
            difficulty=3,
            domains=["computing"],
        ),
        Question(
            id="q2",
            type=QuestionType.GATE,
            variant="counterfactual",
            pair_id="p1",
            prompt="CF Test?",
            answer="No",
            difficulty=4,
            domains=["computing", "physics"],
        ),
    ]


class TestExportForReview:
    def test_csv_export(self, review_questions, tmp_path):
        path = tmp_path / "review.csv"
        export_for_review(review_questions, path, format="csv")
        assert path.exists()
        content = path.read_text()
        assert "reviewer_score" in content
        assert "q1" in content
        assert "q2" in content

    def test_json_export(self, review_questions, tmp_path):
        path = tmp_path / "review.json"
        export_for_review(review_questions, path, format="json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["id"] == "q1"
        assert data[0]["reviewer_score"] == ""

    def test_invalid_format(self, review_questions, tmp_path):
        with pytest.raises(ValueError, match="Unsupported format"):
            export_for_review(review_questions, tmp_path / "x.txt", format="xml")


class TestImportReviews:
    def test_csv_import(self, review_questions, tmp_path):
        path = tmp_path / "review.csv"
        export_for_review(review_questions, path, format="csv")

        # Simulate filling in scores
        lines = path.read_text().splitlines()
        header = lines[0]
        # Find reviewer_score column index
        cols = header.split(",")
        score_idx = cols.index("reviewer_score")
        new_lines = [header]
        for line in lines[1:]:
            parts = line.split(",")
            parts[score_idx] = "4"
            new_lines.append(",".join(parts))
        path.write_text("\n".join(new_lines))

        reviews = import_reviews(path)
        assert len(reviews) == 2
        assert reviews[0]["reviewer_score"] == 4

    def test_json_import(self, tmp_path):
        path = tmp_path / "reviews.json"
        data = [
            {"id": "q1", "reviewer_score": 5},
            {"id": "q2", "reviewer_score": 2},
        ]
        path.write_text(json.dumps(data))
        reviews = import_reviews(path)
        assert len(reviews) == 2
        assert reviews[0]["reviewer_score"] == 5

    def test_empty_score_becomes_none(self, review_questions, tmp_path):
        path = tmp_path / "review.csv"
        export_for_review(review_questions, path, format="csv")
        reviews = import_reviews(path)
        assert reviews[0]["reviewer_score"] is None


class TestInterAnnotatorAgreement:
    def test_perfect_agreement(self):
        ann1 = [{"id": "q1", "reviewer_score": 5}, {"id": "q2", "reviewer_score": 3}]
        ann2 = [{"id": "q1", "reviewer_score": 5}, {"id": "q2", "reviewer_score": 3}]
        result = inter_annotator_agreement([ann1, ann2])
        assert result["kappa"] == 1.0
        assert result["type"] == "cohen"

    def test_no_agreement(self):
        ann1 = [{"id": "q1", "reviewer_score": 1}, {"id": "q2", "reviewer_score": 2}]
        ann2 = [{"id": "q1", "reviewer_score": 2}, {"id": "q2", "reviewer_score": 1}]
        result = inter_annotator_agreement([ann1, ann2])
        assert result["kappa"] < 1.0
        assert result["type"] == "cohen"

    def test_single_annotator(self):
        ann1 = [{"id": "q1", "reviewer_score": 5}]
        result = inter_annotator_agreement([ann1])
        assert result["kappa"] == 1.0

    def test_fleiss_three_annotators(self):
        ann1 = [{"id": "q1", "reviewer_score": 5}, {"id": "q2", "reviewer_score": 3}]
        ann2 = [{"id": "q1", "reviewer_score": 5}, {"id": "q2", "reviewer_score": 3}]
        ann3 = [{"id": "q1", "reviewer_score": 5}, {"id": "q2", "reviewer_score": 3}]
        result = inter_annotator_agreement([ann1, ann2, ann3])
        assert result["kappa"] == 1.0
        assert result["type"] == "fleiss"

    def test_no_common_ids(self):
        ann1 = [{"id": "q1", "reviewer_score": 5}]
        ann2 = [{"id": "q2", "reviewer_score": 3}]
        result = inter_annotator_agreement([ann1, ann2])
        assert result["kappa"] == 0.0


class TestFilterValidated:
    def test_filters_by_score(self, review_questions):
        reviews = [
            {"id": "q1", "reviewer_score": 4},
            {"id": "q2", "reviewer_score": 2},
        ]
        filtered = filter_validated(review_questions, reviews, min_score=3)
        assert len(filtered) == 1
        assert filtered[0].id == "q1"

    def test_none_scores_excluded(self, review_questions):
        reviews = [
            {"id": "q1", "reviewer_score": None},
            {"id": "q2", "reviewer_score": 5},
        ]
        filtered = filter_validated(review_questions, reviews, min_score=3)
        assert len(filtered) == 1
        assert filtered[0].id == "q2"

    def test_all_pass(self, review_questions):
        reviews = [
            {"id": "q1", "reviewer_score": 5},
            {"id": "q2", "reviewer_score": 5},
        ]
        filtered = filter_validated(review_questions, reviews, min_score=1)
        assert len(filtered) == 2
