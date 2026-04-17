import pytest
import math
from knowledge_base import KnowledgeBase
from inference_engine import BayesEngine


@pytest.fixture
def kb():
    return KnowledgeBase("data/knowledge_base.json")


@pytest.fixture
def engine(kb):
    return BayesEngine(kb)


def test_initial_posteriors_equal_priors(engine, kb):
    posteriors = engine.get_posteriors()
    priors = kb.get_priors()
    for diag_id, prior in priors.items():
        assert abs(posteriors[diag_id] - prior) < 1e-9


def test_posteriors_sum_to_one(engine):
    posteriors = engine.get_posteriors()
    assert abs(sum(posteriors.values()) - 1.0) < 1e-9


def test_update_with_positive_symptom(engine):
    before = engine.get_posteriors()["ram_failure"]
    engine.update("blue_screen", True)
    after = engine.get_posteriors()["ram_failure"]
    assert after > before


def test_update_with_negative_symptom(engine):
    before = engine.get_posteriors()["ram_failure"]
    engine.update("blue_screen", False)
    after = engine.get_posteriors()["ram_failure"]
    assert after < before


def test_posteriors_sum_after_updates(engine):
    engine.update("blue_screen", True)
    engine.update("no_power", False)
    posteriors = engine.get_posteriors()
    assert abs(sum(posteriors.values()) - 1.0) < 1e-9


def test_skip_does_not_change_posteriors(engine):
    before = dict(engine.get_posteriors())
    engine.update("blue_screen", None)
    after = engine.get_posteriors()
    for diag_id in before:
        assert abs(before[diag_id] - after[diag_id]) < 1e-9


def test_get_top_n(engine):
    engine.update("blue_screen", True)
    top = engine.get_top_n(3)
    assert len(top) == 3
    assert top[0][1] >= top[1][1] >= top[2][1]


def test_get_top_n_returns_tuples(engine):
    top = engine.get_top_n(3)
    for item in top:
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)


def test_has_confident_diagnosis_false_initially(engine):
    assert engine.has_confident_diagnosis(0.85) is False


def test_has_confident_diagnosis_after_strong_evidence(engine):
    engine.update("date_resets", True)
    engine.update("bios_settings_lost", True)
    assert engine.has_confident_diagnosis(0.85) is True


def test_next_best_question(engine):
    asked = set()
    question = engine.next_best_question(asked)
    assert question is not None
    assert isinstance(question, str)


def test_next_best_question_excludes_asked(engine):
    asked = {"blue_screen", "no_power"}
    question = engine.next_best_question(asked)
    assert question not in asked


def test_next_best_question_returns_none_when_all_asked(engine, kb):
    asked = set(kb.get_symptom_ids())
    question = engine.next_best_question(asked)
    assert question is None
