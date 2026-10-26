# Rain_DB — Análise de Dados e Machine Learning para Diagnóstico de Câncer de Mama

Projeto acadêmico de análise exploratória de dados e classificação com Machine Learning, desenvolvido como parte dos estudos de Ciência de Dados e Inteligência Artificial.

O objetivo é investigar características de tumores mamários e avaliar modelos de classificação capazes de distinguir amostras benignas de malignas a partir do dataset Breast Cancer Wisconsin Diagnostic.

> **Aviso:** este projeto tem finalidade exclusivamente acadêmica e educacional. Os modelos não constituem ferramentas médicas validadas e não devem ser utilizados para diagnóstico, tratamento ou decisões clínicas.

## Objetivos

- Realizar análise exploratória e estatística de dados médicos.
- Identificar correlações entre características e diagnóstico.
- Visualizar distribuições e diferenças entre as classes.
- Treinar e comparar algoritmos de classificação.
- Avaliar os modelos com métricas de classificação e matriz de confusão.
- Investigar a importância das características utilizadas pelo Random Forest.

## Tecnologias utilizadas

- **Python** — linguagem de programação.
- **Pandas** — manipulação e análise de dados.
- **Matplotlib** — criação de gráficos.
- **Seaborn** — visualização estatística.
- **Scikit-learn** — pré-processamento, treinamento e avaliação dos modelos.

## Dataset

O projeto utiliza o **Breast Cancer Wisconsin Diagnostic (WDBC)**, um conjunto de dados com características numéricas extraídas de imagens digitalizadas de amostras de tecido mamário.

Fonte de referência: [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

### Características do conjunto de dados

- **569 registros**.
- **30 características preditoras** após a limpeza.
- **357 amostras benignas**.
- **212 amostras malignas**.
- Variável-alvo: `diagnosis`.

As classes são codificadas da seguinte maneira:

| Valor original | Valor utilizado | Significado |
|---|---:|---|
| `B` | `0` | Benigno |
| `M` | `1` | Maligno |

As colunas `id` e `Unnamed: 32` são removidas durante o pré-processamento. A segunda contém apenas valores ausentes no arquivo utilizado.

## Funcionalidades implementadas

### 1. Análise exploratória de dados (EDA)

O projeto realiza:

- Verificação das dimensões do dataset e dos tipos de dados.
- Identificação de valores ausentes.
- Estatísticas descritivas.
- Contagem de amostras por diagnóstico.
- Análise de correlação entre as características e a variável-alvo.

### 2. Visualização dos dados

O script gera os seguintes gráficos:

- `correlacao_diagnostico.png` — correlação das características com o diagnóstico.
- `radius_mean_diagnostico.png` — distribuição de `radius_mean` por classe.
- `concave_points_mean_diagnostico.png` — distribuição de `concave points_mean` por classe.
- `boxplot_radius_mean.png` — comparação de `radius_mean` entre os diagnósticos.
- `feature_importance.png` — dez características mais importantes para o Random Forest.

Os arquivos são gerados durante a execução do script.

### 3. Pré-processamento

O pré-processamento inclui:

- Remoção de colunas sem utilidade preditiva para o experimento.
- Codificação da variável-alvo.
- Separação dos dados em treino, validação e teste.
- Padronização das características dentro do pipeline da Regressão Logística.

A divisão utiliza amostragem estratificada e `random_state=42` para tornar o experimento reproduzível nas mesmas condições.

### 4. Modelos de Machine Learning

Foram implementados dois classificadores:

**Regressão Logística**

Utiliza `StandardScaler` para padronizar as características antes do treinamento. O modelo é avaliado considerando seu desempenho na identificação da classe maligna.

**Random Forest**

Utiliza um conjunto de árvores de decisão, configurado com 200 estimadores, para realizar a classificação e investigar a importância relativa das características.

### 5. Avaliação dos modelos

Os modelos são comparados pelo recall da classe maligna no conjunto de validação. O modelo com maior recall é selecionado para a avaliação final no conjunto de teste.

As métricas utilizadas incluem:

- Acurácia (*accuracy*).
- Precisão (*precision*).
- Sensibilidade (*recall*).
- F1-score.
- Matriz de confusão.

## Resultados experimentais

Na execução registrada, a Regressão Logística foi selecionada com base no recall da validação.

| Métrica no conjunto de teste | Resultado |
|---|---:|
| Acurácia | 97,37% |
| Precisão — classe maligna | 97,56% |
| Recall — classe maligna | 95,24% |
| F1-score — classe maligna | 96,39% |
| Amostras de teste | 114 |

### Matriz de confusão

| Diagnóstico real | Previsto benigno | Previsto maligno |
|---|---:|---:|
| Benigno | 71 | 1 |
| Maligno | 2 | 40 |

O modelo classificou corretamente 111 das 114 amostras do conjunto de teste. Entretanto, dois casos malignos foram classificados como benignos e um caso benigno foi classificado como maligno.

Esses erros demonstram a importância de analisar as métricas por classe, em vez de avaliar o desempenho apenas pela acurácia.

**Importante:** os resultados correspondem a uma única divisão do dataset. Eles não demonstram desempenho clínico nem garantem resultados equivalentes em outras amostras ou populações.

## Importância das características

A análise do Random Forest identificou as seguintes características como as dez mais importantes na execução registrada:

| Característica | Importância |
|---|---:|
| `perimeter_worst` | 15,63% |
| `concave points_worst` | 11,85% |
| `area_worst` | 11,04% |
| `concave points_mean` | 10,61% |
| `radius_mean` | 6,47% |
| `radius_worst` | 5,98% |
| `perimeter_mean` | 5,34% |
| `concavity_mean` | 4,80% |
| `area_mean` | 3,99% |
| `concavity_worst` | 3,79% |

As quatro primeiras características representam aproximadamente 44,12% da importância total atribuída pelo Random Forest.

Esses valores indicam a importância relativa das variáveis nas divisões das árvores do modelo. Não representam causalidade nem devem ser interpretados como explicações diretas das previsões da Regressão Logística.

## Estrutura do projeto

```text
Rain_DB/
├── data/
│   └── data.csv
├── main.py
├── .gitignore
├── README.md
├── correlacao_diagnostico.png
├── radius_mean_diagnostico.png
├── concave_points_mean_diagnostico.png
├── boxplot_radius_mean.png
└── feature_importance.png
```

Os arquivos PNG são resultados gerados pelo script. A estrutura acima representa a organização esperada do projeto; a presença de cada imagem na raiz depende de sua geração e de ela ter sido incluída no repositório.

## Como executar o projeto

### Pré-requisitos

- Python 3.12 ou versão compatível com as bibliotecas utilizadas.
- Git.
- Dataset disponível no caminho `data/data.csv`.

### 1. Clonar o repositório

```bash
git clone https://github.com/DaviC18/Rain_DB.git
cd Rain_DB
```

### 2. Criar e ativar um ambiente virtual

No Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
python -m pip install --upgrade pip
python -m pip install pandas matplotlib seaborn scikit-learn
```

### 4. Verificar o dataset

Confirme que o arquivo está no caminho:

```text
data/data.csv
```

O arquivo deve conter as características do dataset WDBC e a coluna `diagnosis`.

### 5. Executar o projeto

```bash
python main.py
```

O script realizará a análise exploratória, gerará os gráficos, treinará os modelos, comparará o recall na validação e exibirá as métricas da avaliação final.

Os gráficos serão salvos no diretório de trabalho da execução.

## Limitações e considerações

- O projeto utiliza um dataset relativamente pequeno e específico.
- A avaliação apresentada utiliza uma única divisão entre treino, validação e teste.
- O recall da classe maligna é priorizado na seleção do modelo, mas não representa, isoladamente, segurança clínica.
- A importância das características do Random Forest não estabelece relações causais.
- Não foi realizada validação clínica, externa ou prospectiva.
- O projeto é experimental e não substitui profissionais de saúde nem procedimentos diagnósticos validados.

## Possíveis melhorias futuras

- Implementar validação cruzada estratificada.
- Avaliar curvas ROC e Precision-Recall.
- Investigar diferentes limiares de classificação.
- Aplicar SHAP para complementar a interpretabilidade.
- Comparar os coeficientes da Regressão Logística após a padronização.
- Organizar as dependências em um arquivo `requirements.txt`.
- Adicionar testes automatizados e melhorar a configuração do caminho do dataset.
- Documentar a análise crítica e os resultados em um relatório técnico.

## Contexto acadêmico

Projeto desenvolvido para fins de aprendizado em análise de dados, estatística aplicada e Machine Learning, com foco em classificação supervisionada e interpretação de resultados.

## Autor

**Davi Valério Rocha de Souza Curty**

- GitHub: [@DaviC18](https://github.com/DaviC18)
- Repositório: [Rain_DB](https://github.com/DaviC18/Rain_DB)
