import os


class CLI:
    CONFIDENCE_THRESHOLD = 0.85

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def show_welcome(self):
        self.clear_screen()
        print("══════════════════════════════════════════════════")
        print("  TechDoc - Sistema Especialista em")
        print("  Diagnóstico de Problemas de Computador")
        print("══════════════════════════════════════════════════")
        print()
        print("  Responda as perguntas sobre os sintomas do")
        print("  seu computador e o sistema irá diagnosticar")
        print("  o problema mais provável.")
        print()
        print("  Opções de resposta:")
        print("    [S] Sim    [N] Não")
        print("    [P] Pular  [Q] Sair")
        print()
        print("══════════════════════════════════════════════════")
        print()

    def ask_symptom(self, question: str, current: int, total: int) -> str | None:
        print(f"  Pergunta {current}/{total}:")
        print(f"    {question}")
        while True:
            answer = input("    > ").strip().lower()
            if answer in ("s", "sim", "y", "yes"):
                return "yes"
            elif answer in ("n", "nao", "não", "no"):
                return "no"
            elif answer in ("p", "pular", "skip"):
                return "skip"
            elif answer in ("q", "quit", "sair"):
                return "quit"
            else:
                print("    Resposta inválida. Use [S]im, [N]ão, [P]ular ou [Q]Sair.")

    def show_current_leader(self, name: str, probability: float):
        print(f"    Diagnóstico líder: {name} ({probability:.1%})")
        print()

    def show_results(self, results: list[tuple[str, str, float, list[str]]]):
        print()
        print("══════════════════════════════════════════════════")
        print("  RESULTADO DO DIAGNÓSTICO")
        print("══════════════════════════════════════════════════")
        print()
        for i, (diag_id, name, prob, solutions) in enumerate(results, 1):
            print(f"  {i}. {name} — {prob:.1%}")
            for solution in solutions:
                print(f"     → {solution}")
            print()

    def ask_restart(self) -> bool:
        answer = input("  Deseja iniciar novo diagnóstico? [S/N] ").strip().lower()
        return answer in ("s", "sim", "y", "yes")

    def show_goodbye(self):
        print()
        print("  Obrigado por usar o TechDoc!")
        print("  Consulte um técnico se o problema persistir.")
        print()
