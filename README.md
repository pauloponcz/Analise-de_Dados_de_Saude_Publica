# Análise de Fatores de Risco para AVC

Este projeto realiza uma análise de dados de saúde pública para identificar fatores de risco relacionados ao AVC (Acidente Vascular Cerebral) utilizando um dataset público de previsão de AVC.

## Estrutura do Projeto

- **Limpeza de Dados**: O dataset é pré-processado para tratar valores ausentes e transformar variáveis categóricas em numéricas usando técnicas de codificação.
- **Análise de Fatores de Risco**: São gerados gráficos com a biblioteca `Seaborn` para explorar a relação entre hipertensão, doenças cardíacas e a ocorrência de AVC.
- **Distribuição Etária**: Um histograma da distribuição etária dos pacientes que sofreram AVC é criado utilizando `Matplotlib`.
- **Modelagem Simples**: Implementação de dois modelos de classificação binária (Regressão Logística e Random Forest) para prever a probabilidade de um paciente sofrer AVC.

## Como executar o projeto

### Pré-requisitos

As seguintes bibliotecas Python são necessárias para executar o projeto:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Instalação das dependências

Use o seguinte comando para instalar todas as dependências:

```bash
pip istall -r .\requirements.txt
