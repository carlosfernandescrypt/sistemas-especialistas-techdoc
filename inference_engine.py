import math


class BayesEngine:
    def __init__(self, knowledge_base):
        self._kb = knowledge_base
        self._posteriors = dict(knowledge_base.get_priors())

    def get_posteriors(self) -> dict[str, float]:
        return dict(self._posteriors)

    def update(self, symptom_id: str, observed: bool | None) -> None:
        if observed is None:
            return

        new_posteriors = {}
        for diag_id, prior in self._posteriors.items():
            likelihood = self._kb.get_likelihood(symptom_id, diag_id)
            if observed:
                new_posteriors[diag_id] = likelihood * prior
            else:
                new_posteriors[diag_id] = (1 - likelihood) * prior

        total = sum(new_posteriors.values())
        if total > 0:
            self._posteriors = {k: v / total for k, v in new_posteriors.items()}

    def get_top_n(self, n: int) -> list[tuple[str, float]]:
        sorted_diags = sorted(
            self._posteriors.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_diags[:n]

    def has_confident_diagnosis(self, threshold: float) -> bool:
        if not self._posteriors:
            return False
        max_prob = max(self._posteriors.values())
        return max_prob >= threshold

    def next_best_question(self, asked: set[str]) -> str | None:
        remaining = [s for s in self._kb.get_symptom_ids() if s not in asked]
        if not remaining:
            return None

        best_symptom = None
        best_gain = -1.0

        current_entropy = self._entropy(list(self._posteriors.values()))

        for symptom_id in remaining:
            p_symptom = sum(
                self._kb.get_likelihood(symptom_id, d) * self._posteriors[d]
                for d in self._posteriors
            )
            p_symptom = max(min(p_symptom, 0.999), 0.001)

            posteriors_yes = {}
            posteriors_no = {}
            for d, p in self._posteriors.items():
                lk = self._kb.get_likelihood(symptom_id, d)
                posteriors_yes[d] = lk * p
                posteriors_no[d] = (1 - lk) * p

            total_yes = sum(posteriors_yes.values())
            total_no = sum(posteriors_no.values())

            if total_yes > 0:
                vals_yes = [v / total_yes for v in posteriors_yes.values()]
            else:
                vals_yes = list(posteriors_yes.values())

            if total_no > 0:
                vals_no = [v / total_no for v in posteriors_no.values()]
            else:
                vals_no = list(posteriors_no.values())

            expected_entropy = (
                p_symptom * self._entropy(vals_yes)
                + (1 - p_symptom) * self._entropy(vals_no)
            )
            gain = current_entropy - expected_entropy

            if gain > best_gain:
                best_gain = gain
                best_symptom = symptom_id

        return best_symptom

    def _entropy(self, probs: list[float]) -> float:
        return -sum(p * math.log2(p) for p in probs if p > 0)
