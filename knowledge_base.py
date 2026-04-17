import json


class KnowledgeBase:
    DEFAULT_LIKELIHOOD = 0.05

    def __init__(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._diagnostics = data["diagnostics"]
        self._symptoms = data["symptoms"]
        self._likelihood_map = {}
        for symptom in self._symptoms:
            self._likelihood_map[symptom["id"]] = symptom["likelihoods"]
        self._diagnostic_map = {d["id"]: d for d in self._diagnostics}
        self._symptom_map = {s["id"]: s for s in self._symptoms}

    def get_diagnostics(self) -> list[dict]:
        return self._diagnostics

    def get_symptoms(self) -> list[dict]:
        return self._symptoms

    def get_likelihood(self, symptom_id: str, diagnostic_id: str) -> float:
        likelihoods = self._likelihood_map.get(symptom_id, {})
        return likelihoods.get(diagnostic_id, self.DEFAULT_LIKELIHOOD)

    def get_priors(self) -> dict[str, float]:
        raw = {d["id"]: d["prior"] for d in self._diagnostics}
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def get_symptom_ids(self) -> list[str]:
        return [s["id"] for s in self._symptoms]

    def get_solutions(self, diagnostic_id: str) -> list[str]:
        return self._diagnostic_map[diagnostic_id]["solutions"]

    def get_question(self, symptom_id: str) -> str:
        return self._symptom_map[symptom_id]["question"]

    def get_diagnostic_name(self, diagnostic_id: str) -> str:
        return self._diagnostic_map[diagnostic_id]["name"]
