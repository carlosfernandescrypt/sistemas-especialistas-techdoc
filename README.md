# TechDoc — Sistema Especialista em Diagnóstico de Problemas de Computador

Sistema especialista que utiliza inferência Bayesiana (Naive Bayes) para diagnosticar problemas de computador. Desenvolvido como trabalho acadêmico (AT1) de Inteligência Artificial.

## Funcionalidades

- 22 diagnósticos de problemas comuns de computador
- 28 sintomas mapeados com probabilidades
- Motor de inferência Naive Bayes implementado do zero
- Seleção inteligente de perguntas por ganho de informação
- Interface interativa no terminal

## Requisitos

- Python 3.10+

## Como Executar

```bash
python main.py
```

## Como Rodar os Testes

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Arquitetura

```
┌─────────────────┐
│   Interface CLI  │  ← perguntas e respostas
├─────────────────┤
│ Motor Inferência │  ← Naive Bayes
├─────────────────┤
│Base Conhecimento │  ← JSON com diagnósticos/sintomas
└─────────────────┘
```

## Estrutura

- `main.py` — ponto de entrada
- `inference_engine.py` — algoritmo Naive Bayes com seleção por ganho de informação
- `knowledge_base.py` — carregamento e acesso à base de conhecimento
- `cli.py` — interface do terminal
- `data/knowledge_base.json` — base de conhecimento (diagnósticos e sintomas)
- `tests/` — testes unitários
