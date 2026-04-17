# TechDoc — Relatório do Sistema Especialista para Diagnóstico de Problemas de Computador

## 1. Introdução

### 1.1 Definição do Problema

Problemas de computador são uma das queixas mais frequentes em ambientes domésticos e corporativos. A diversidade de possíveis causas — desde falhas de hardware até conflitos de software — torna o diagnóstico uma tarefa que exige conhecimento técnico especializado, nem sempre acessível ao usuário comum.

O TechDoc é um sistema especialista que auxilia usuários a diagnosticar problemas em seus computadores por meio de uma entrevista interativa. O sistema faz perguntas sobre os sintomas observados e, com base nas respostas, calcula a probabilidade de cada possível diagnóstico, apresentando os mais prováveis junto com soluções recomendadas.

### 1.2 Relevância

Sistemas especialistas são uma das aplicações mais clássicas e bem-sucedidas da Inteligência Artificial. Desde o MYCIN (1976), que diagnosticava infecções bacterianas, até sistemas modernos de suporte à decisão, a ideia de codificar o conhecimento de um especialista em regras computacionais tem se mostrado eficaz em domínios bem definidos.

O diagnóstico de computadores é um domínio ideal para um sistema especialista porque:

- O conjunto de problemas é finito e bem catalogado
- Os sintomas são observáveis e relatáveis pelo usuário
- Existe uma relação probabilística clara entre sintomas e diagnósticos
- O conhecimento de especialistas pode ser formalizado em regras e probabilidades

## 2. Fundamentação Teórica

### 2.1 Sistemas Especialistas

Um sistema especialista é um programa de computador que utiliza conhecimento especializado para resolver problemas em um domínio específico, simulando o raciocínio de um especialista humano (Russell & Norvig, 2021). Seus componentes fundamentais são:

- **Base de conhecimento:** armazena fatos e regras sobre o domínio
- **Motor de inferência:** aplica o conhecimento para derivar conclusões
- **Interface com o usuário:** permite a interação para coleta de dados e apresentação de resultados

O TechDoc implementa essa arquitetura clássica em três camadas: uma base de conhecimento em JSON, um motor de inferência Bayesiano e uma interface de linha de comando (CLI).

### 2.2 Representação do Conhecimento

A representação do conhecimento é o processo de codificar informações sobre o mundo de forma que um sistema computacional possa utilizá-las para resolver problemas (Brachman & Levesque, 2004).

Neste projeto, o conhecimento é representado por meio de **relações probabilísticas entre sintomas e diagnósticos**. Cada diagnóstico possui:

- Uma **probabilidade a priori** (`prior`): a probabilidade de ocorrência do problema antes de qualquer observação, baseada em frequências estimadas de atendimentos técnicos
- Uma lista de **soluções recomendadas**: passos que o usuário pode seguir para resolver o problema

Cada sintoma possui:

- Uma **pergunta** em linguagem natural
- Um mapa de **verossimilhanças** (`likelihoods`): para cada diagnóstico relevante, a probabilidade de o sintoma estar presente dado que aquele diagnóstico é a causa real

Essa representação é equivalente a uma rede Bayesiana simplificada (Naive Bayes), onde os sintomas são condicionalmente independentes dado o diagnóstico.

### 2.3 Teorema de Bayes

O Teorema de Bayes é o fundamento matemático do motor de inferência do TechDoc. Formulado por Thomas Bayes no século XVIII, ele descreve como atualizar a probabilidade de uma hipótese à luz de novas evidências (Bayes, 1763):

$$P(D|S) = \frac{P(S|D) \cdot P(D)}{P(S)}$$

Onde:

- $P(D|S)$ é a **probabilidade posterior** — a probabilidade do diagnóstico $D$ dado que o sintoma $S$ foi observado
- $P(S|D)$ é a **verossimilhança** — a probabilidade do sintoma $S$ aparecer dado o diagnóstico $D$
- $P(D)$ é a **probabilidade a priori** — a probabilidade inicial do diagnóstico antes da observação
- $P(S)$ é a **evidência** — a probabilidade total do sintoma (constante de normalização)

### 2.4 Naive Bayes

O classificador Naive Bayes é uma simplificação que assume **independência condicional entre os sintomas** dado o diagnóstico (Mitchell, 1997). Isso significa que a presença ou ausência de um sintoma não afeta a probabilidade de outro sintoma, dado que já sabemos o diagnóstico.

Embora essa suposição raramente seja verdadeira na prática (por exemplo, tela azul e reinício aleatório frequentemente coocorrem em problemas de RAM), o Naive Bayes tem se mostrado surpreendentemente eficaz mesmo quando a suposição de independência é violada (Domingos & Pazzani, 1997).

Para múltiplos sintomas observados $S_1, S_2, ..., S_n$, a fórmula se torna:

$$P(D|S_1, S_2, ..., S_n) \propto P(D) \cdot \prod_{i=1}^{n} P(S_i|D)$$

No TechDoc, a atualização é feita incrementalmente: a cada resposta do usuário, as probabilidades posteriores são recalculadas e normalizadas para somar 1.

### 2.5 Tratamento de Incerteza

O tratamento de incerteza é um aspecto central do TechDoc. O sistema lida com incerteza em três níveis:

1. **Incerteza nas probabilidades a priori:** os valores de `prior` são estimativas baseadas em frequências observadas, não valores exatos. O sistema normaliza essas probabilidades para garantir consistência matemática.

2. **Incerteza nas verossimilhanças:** os valores de `likelihood` representam a relação probabilística entre sintomas e diagnósticos. Diagnósticos não explicitamente listados para um sintoma recebem uma verossimilhança default de 0.05, representando a baixa (mas não nula) probabilidade de o sintoma ocorrer por uma causa não mapeada.

3. **Incerteza nas respostas do usuário:** o sistema oferece a opção "Pular" para quando o usuário não sabe responder. Nesse caso, as probabilidades não são atualizadas para aquele sintoma, evitando a introdução de informação incorreta.

O critério de parada por confiança (probabilidade posterior ≥ 85%) também é uma forma de tratamento de incerteza: o sistema só apresenta o diagnóstico como confiável quando a evidência acumulada é suficiente.

### 2.6 Ganho de Informação e Entropia

Para otimizar o processo de diagnóstico, o TechDoc utiliza o conceito de **ganho de informação** (Information Gain) para selecionar a próxima pergunta a ser feita (Quinlan, 1986). Em vez de perguntar os sintomas em ordem fixa, o sistema escolhe o sintoma que mais reduz a incerteza sobre o diagnóstico.

A **entropia** mede a incerteza de uma distribuição de probabilidades:

$$H(X) = -\sum_{i} P(x_i) \cdot \log_2 P(x_i)$$

O **ganho de informação** de um sintoma $S$ é a redução esperada na entropia:

$$IG(S) = H(D) - [P(S) \cdot H(D|S) + P(\neg S) \cdot H(D|\neg S)]$$

O sistema calcula o ganho de informação para cada sintoma ainda não perguntado e seleciona aquele com maior ganho. Isso resulta em diagnósticos mais rápidos e precisos, pois as perguntas mais discriminativas são feitas primeiro.

## 3. Metodologia

### 3.1 Escolha do Domínio

O domínio de diagnóstico de problemas de computador foi escolhido por reunir características ideais para um sistema especialista acadêmico:

- Problemas bem definidos e catalogáveis
- Sintomas observáveis sem equipamentos especiais
- Público-alvo amplo (qualquer usuário de computador)
- Conhecimento especialista disponível na literatura técnica

### 3.2 Escolha das Técnicas

**Naive Bayes** foi escolhido como técnica de inferência pelos seguintes motivos:

- Fundamentação teórica sólida no Teorema de Bayes
- Implementação transparente e didática (sem dependências externas)
- Atualização incremental natural (a cada resposta do usuário)
- Tratamento nativo de incerteza via probabilidades
- Eficácia comprovada mesmo com a suposição de independência

Alternativas consideradas e descartadas:

| Técnica | Motivo do descarte |
|---|---|
| Regras de produção (SE-ENTÃO) | Não trata incerteza nativamente, regras se tornam complexas com muitos diagnósticos |
| Lógica Fuzzy | Mais adequada para variáveis contínuas (ex: temperatura), menos natural para sintomas binários |
| Redes Bayesianas completas | Complexidade desnecessária para o escopo; exigiriam definir dependências entre todos os sintomas |
| Fatores de Certeza (MYCIN) | Menos fundamentação matemática que Bayes; ad hoc na combinação de fatores |

**Ganho de informação** foi escolhido para seleção de perguntas por:

- Minimizar o número de perguntas necessárias para um diagnóstico
- Fundamentação teórica na Teoria da Informação de Shannon
- Mesmo princípio usado em árvores de decisão (ID3/C4.5)

### 3.3 Construção da Base de Conhecimento

A base de conhecimento foi construída com 22 diagnósticos e 28 sintomas, cobrindo as categorias:

- **Hardware:** RAM, HD/SSD, fonte, placa-mãe, GPU, cooler, cabos, monitor, periféricos, ventilação, bateria CMOS
- **Software:** vírus/malware, drivers, sistema operacional, conflitos de software, disco cheio, partição corrompida, registro do Windows, problemas de boot
- **Configuração:** BIOS desatualizada, placa de rede

As probabilidades a priori foram estimadas com base na frequência relativa de cada tipo de problema em cenários típicos de suporte técnico. As verossimilhanças foram definidas considerando a correlação clínica entre cada sintoma e cada diagnóstico.

### 3.4 Ferramentas Utilizadas

| Ferramenta | Uso |
|---|---|
| Python 3.12 | Linguagem de implementação |
| JSON | Formato da base de conhecimento |
| pytest | Framework de testes unitários |
| Biblioteca padrão (json, math, os) | Manipulação de dados, cálculos, interface |

A escolha por não utilizar bibliotecas externas de IA (como scikit-learn ou pgmpy) foi deliberada: o objetivo acadêmico é demonstrar a compreensão dos fundamentos implementando o algoritmo do zero.

## 4. Arquitetura do Sistema

```
┌─────────────────────────┐
│     Interface CLI        │  ← Interação com o usuário
│       (cli.py)           │
├─────────────────────────┤
│    Motor de Inferência   │  ← Naive Bayes + Ganho de Informação
│  (inference_engine.py)   │
├─────────────────────────┤
│   Base de Conhecimento   │  ← 22 diagnósticos, 28 sintomas
│  (knowledge_base.py)     │
│  (knowledge_base.json)   │
└─────────────────────────┘
```

### 4.1 Fluxo de Execução

1. O sistema carrega a base de conhecimento do arquivo JSON
2. Inicializa as probabilidades posteriores com os valores a priori
3. Calcula qual sintoma tem maior ganho de informação
4. Apresenta a pergunta ao usuário
5. Atualiza as probabilidades com base na resposta (Sim/Não/Pular)
6. Repete até: diagnóstico confiável (≥85%), todas as perguntas feitas, ou usuário sair
7. Apresenta os 3 diagnósticos mais prováveis com soluções

### 4.2 Separação de Responsabilidades

- **`knowledge_base.py`** — Encapsula o acesso aos dados, isola o formato de armazenamento
- **`inference_engine.py`** — Implementa a lógica de inferência, independente da interface
- **`cli.py`** — Cuida exclusivamente da apresentação e coleta de dados
- **`main.py`** — Orquestra os componentes, gerencia o ciclo de vida da aplicação

## 5. Resultados

### 5.1 Testes Automatizados

O sistema possui 21 testes unitários cobrindo:

- **Base de conhecimento (8 testes):** carregamento, acesso a diagnósticos, sintomas, verossimilhanças, priors, soluções
- **Motor de inferência (13 testes):** inicialização, atualização Bayesiana (positiva, negativa, skip), normalização, ranking, diagnóstico confiante, seleção de perguntas por ganho de informação

Todos os 21 testes passam com sucesso.

### 5.2 Demonstração de Funcionamento

**Cenário 1: Problema de RAM**

Ao responder "Sim" para tela azul, bips ao ligar e programas travando, e "Não" para pop-ups inesperados e disco trabalhando sem parar, o sistema converge para "Falha na Memória RAM" com alta confiança, sugerindo reencaixar pentes, testar individualmente e usar MemTest86.

**Cenário 2: Vírus/Malware**

Ao responder "Sim" para lentidão extrema, pop-ups inesperados e uso alto de CPU, o sistema converge rapidamente para "Vírus/Malware", sugerindo antivírus, Malwarebytes e verificação de programas recentes.

**Cenário 3: Bateria CMOS**

Ao responder "Sim" para data resetando e configurações da BIOS perdidas, o sistema atinge confiança ≥85% em apenas 2 perguntas, demonstrando a eficácia do ganho de informação na seleção de perguntas discriminativas.

### 5.3 Eficácia do Ganho de Informação

A seleção inteligente de perguntas reduz significativamente o número de perguntas necessárias para um diagnóstico. Em problemas com sintomas altamente discriminativos (como bateria CMOS ou malware), o sistema alcança confiança em 2-5 perguntas, em vez das 28 possíveis.

## 6. Conclusão

O TechDoc demonstra a aplicação prática dos fundamentos da Inteligência Artificial em um sistema especialista funcional. O projeto cobriu:

- **Representação do conhecimento:** base de dados estruturada com relações probabilísticas entre sintomas e diagnósticos
- **Tratamento de incerteza:** inferência Bayesiana com atualização incremental e normalização de probabilidades
- **Resolução de problemas:** seleção inteligente de perguntas por ganho de informação, convergência eficiente para diagnósticos
- **Aprendizagem de máquina:** o Naive Bayes é um classificador probabilístico que serve como ponte entre sistemas especialistas clássicos e técnicas modernas de machine learning

O sistema é extensível: novos diagnósticos e sintomas podem ser adicionados editando apenas o arquivo JSON, sem alterar o código.

## Referências

- BAYES, T. An Essay towards solving a Problem in the Doctrine of Chances. *Philosophical Transactions of the Royal Society of London*, v. 53, p. 370-418, 1763.
- BRACHMAN, R. J.; LEVESQUE, H. J. *Knowledge Representation and Reasoning*. Morgan Kaufmann, 2004.
- DOMINGOS, P.; PAZZANI, M. On the Optimality of the Simple Bayesian Classifier under Zero-One Loss. *Machine Learning*, v. 29, p. 103-130, 1997.
- MITCHELL, T. M. *Machine Learning*. McGraw-Hill, 1997.
- QUINLAN, J. R. Induction of Decision Trees. *Machine Learning*, v. 1, p. 81-106, 1986.
- RUSSELL, S.; NORVIG, P. *Artificial Intelligence: A Modern Approach*. 4th ed. Pearson, 2021.
- SHORTLIFFE, E. H. *Computer-Based Medical Consultations: MYCIN*. Elsevier, 1976.
