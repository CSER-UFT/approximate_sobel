# DSE GRASP - Filtro de Sobel com Somadores Aproximados

Este projeto realiza a **Exploração do Espaço de Projeto (DSE - Design Space Exploration)** utilizando a meta-heurística **GRASP** (*Greedy Randomized Adaptive Search Procedure*) aplicada ao algoritmo de detecção de bordas **Filtro de Sobel**. 

O objetivo é substituir os 9 somadores exatos do kernel de Sobel por somadores aproximados (*Trunc-0* e *Trunc-1* de 1 a 5 bits), minimizando o consumo de área em hardware (LUTs em FPGA via **Xilinx Vitis HLS**) enquanto respeita limites de degradação de qualidade de imagem medidos via **SSIM** (*Structural Similarity Index*).

---

## Estrutura Principal

- `dse_grasp_sobel.ipynb`: Notebook principal contendo todo o fluxo de execução do DSE, avaliação de métricas, geração de gráficos e scripts de síntese.
- `grasp/`: Pacote Python com o núcleo da meta-heurística GRASP (`construcao.py`, `busca_local.py`, `avaliador.py`, `main_grasp.py`).
- `scripts/`: Scripts auxiliares de análise, plotagem de métricas, inspeção visual e automação de síntese no Vitis HLS.
- `results/`: Tabelas consolidadas (`.csv`), gráficos vetoriais (`.pdf`) e relatórios de síntese do Vitis HLS (`.rpt` e `.xml`).
- `src/` e `templates/`: Implementações em C/C++ dos somadores exato e aproximados (`add_exact`, `add_trunc0_1..5`, `add_trunc1_1..5`) e do filtro de Sobel (`sobel.cpp`, `template_sobel.cpp`).
- `image_dataset-main/`: Dataset de imagens utilizado para treino e teste das configurações.

---

## Pré-requisitos

- **Python 3.8+**
- **G++** (para compilação rápida do binário `sobel` durante a avaliação no GRASP)
- **Xilinx Vitis HLS** (opcional, necessário apenas para rodar novamente a síntese de hardware em FPGA)

---

## Como Executar

### 1. Configurar o ambiente Python

Crie/ative um ambiente virtual e instale as dependências necessárias:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Compilar o executável base do Sobel

Antes de rodar a exploração no Python, compile o código C++ do filtro de Sobel:

```bash
g++ -O3 -fopenmp sobel.cpp -o sobel
```

### 3. Executar a Exploração (DSE com GRASP)

A maneira principal de executar os experimentos, visualizar os resultados de área vs. erro (SSIM) e gerar os gráficos é através do Jupyter Notebook:

```bash
jupyter notebook dse_grasp_sobel.ipynb
```

Você também pode executar os scripts auxiliares diretamente pelo terminal:

```bash
# Calcular métricas de erro (ED, ER, MED, MRED) dos somadores aproximados
python3 scripts/erro_somador.py

# Comparar resultados do GRASP contra busca aleatória (Monte Carlo)
python3 scripts/comparar_grasp_vs_aleatorio.py

# Plotar métricas de detecção de bordas por threshold
python3 scripts/plotar_metricas_th.py --th all

# Inspecionar visualmente as bordas geradas para uma imagem do conjunto de teste
python3 scripts/inspecionar_bordas.py --id 75
```

### 4. Executar a Síntese no Vitis HLS (Opcional)

Caso possua o **Xilinx Vitis HLS** instalado para extrair novamente os relatórios de LUTs, FFs e Clock:

```bash
# 1. Sintetizar os somadores individuais e extrair tabela de LUTs
vitis_hls -f scripts/run_hls.tcl
python3 scripts/parser_adders.py

# 2. Sintetizar o design completo do Sobel com as soluções do GRASP
./scripts/executar_sintese_vitis_geometrico.sh

# 3. Sintetizar os 100 designs da amostragem aleatória (Monte Carlo)
./scripts/executar_sintese_vitis_aleatorio.sh
```
