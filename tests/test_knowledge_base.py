import pytest
from knowledge_base import KnowledgeBase


@pytest.fixture
def kb():
    return KnowledgeBase("data/knowledge_base.json")


def test_load_diagnostics(kb):
    diagnostics = kb.get_diagnostics()
    assert len(diagnostics) >= 22
    assert all("id" in d for d in diagnostics)
    assert all("name" in d for d in diagnostics)
    assert all("prior" in d for d in diagnostics)
    assert all("solutions" in d for d in diagnostics)


def test_load_symptoms(kb):
    symptoms = kb.get_symptoms()
    assert len(symptoms) >= 25
    assert all("id" in s for s in symptoms)
    assert all("question" in s for s in symptoms)
    assert all("likelihoods" in s for s in symptoms)


def test_get_likelihood_existing(kb):
    likelihood = kb.get_likelihood("blue_screen", "ram_failure")
    assert likelihood == 0.85


def test_get_likelihood_default(kb):
    likelihood = kb.get_likelihood("blue_screen", "disk_full")
    assert likelihood == 0.05


def test_get_priors(kb):
    priors = kb.get_priors()
    assert isinstance(priors, dict)
    assert "ram_failure" in priors
    assert abs(sum(priors.values()) - 1.0) < 0.01


def test_get_symptom_ids(kb):
    ids = kb.get_symptom_ids()
    assert "blue_screen" in ids
    assert "no_power" in ids


def test_get_solutions(kb):
    solutions = kb.get_solutions("ram_failure")
    assert len(solutions) >= 2
    assert isinstance(solutions[0], str)


def test_get_question(kb):
    question = kb.get_question("blue_screen")
    assert "tela azul" in question.lower()
