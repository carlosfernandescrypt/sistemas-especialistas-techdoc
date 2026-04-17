from knowledge_base import KnowledgeBase
from inference_engine import BayesEngine
from cli import CLI


def run_diagnosis(kb: KnowledgeBase, engine: BayesEngine, cli: CLI):
    total_symptoms = len(kb.get_symptom_ids())
    asked = set()
    current = 0

    while True:
        symptom_id = engine.next_best_question(asked)
        if symptom_id is None:
            break

        current += 1
        question = kb.get_question(symptom_id)
        answer = cli.ask_symptom(question, current, total_symptoms)

        if answer == "quit":
            break

        asked.add(symptom_id)

        if answer == "yes":
            engine.update(symptom_id, True)
        elif answer == "no":
            engine.update(symptom_id, False)
        else:
            engine.update(symptom_id, None)

        top = engine.get_top_n(1)
        if top:
            diag_id, prob = top[0]
            name = kb.get_diagnostic_name(diag_id)
            cli.show_current_leader(name, prob)

        if engine.has_confident_diagnosis(CLI.CONFIDENCE_THRESHOLD):
            break

    top3 = engine.get_top_n(3)
    results = []
    for diag_id, prob in top3:
        name = kb.get_diagnostic_name(diag_id)
        solutions = kb.get_solutions(diag_id)
        results.append((diag_id, name, prob, solutions))
    cli.show_results(results)


def main():
    kb = KnowledgeBase("data/knowledge_base.json")
    cli = CLI()

    while True:
        engine = BayesEngine(kb)
        cli.show_welcome()
        run_diagnosis(kb, engine, cli)
        if not cli.ask_restart():
            break

    cli.show_goodbye()


if __name__ == "__main__":
    main()
