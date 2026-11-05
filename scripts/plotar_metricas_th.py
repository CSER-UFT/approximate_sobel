#!/usr/bin/env python3
"""
plotar_metricas_th.py

Script parametrizado para execução do benchmark e geração de gráficos boxplot
isolados por threshold (Treino e Teste) para comparação das métricas de erro SSIM:
- ssim_medio (Média Aritmética)
- ssim_mediana (Mediana - Robusta a Outliers)
- ssim_mediageometrica (Média Geométrica)

Gera e salva para cada threshold:
1. Boxplot do Treino (70 imagens de otimização)
2. Boxplot do Teste (30 imagens inéditas de generalização)
3. Boxplot Comparativo Lado a Lado (Treino vs Teste com a mesma escala no eixo Y)
4. Tabela consolidada de métricas numéricas em CSV

Uso:
    python3 plotar_metricas_th.py --th 0.15
    python3 plotar_metricas_th.py --th 0.05 0.10 0.15
    python3 plotar_metricas_th.py --th all
    python3 plotar_metricas_th.py --th 0.15 --iter 30 --show
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Garante importação do pacote grasp na raiz e de comparador_metricas em scripts/
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
for p in (SCRIPTS_DIR, PROJ_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from comparador_metricas import (
    executar_benchmark_metricas,
    gerar_par_boxplots,
    gerar_grafico_boxplot_isolado,
    gerar_grafico_comparativo_lado_a_lado,
    imprimir_guia_escolha_tcc,
    CORES_METRICAS,
    NOMES_LEGIVEIS,
    ROTULOS_EXIBICAO
)
from grasp.avaliador import DATASET_TREINO_PADRAO, DATASET_TESTE_PADRAO

THRESHOLDS_PADRAO = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def formatar_tabela_resumo(df_th):
    """Retorna uma representação limpa em string para o terminal."""
    colunas = [
        ('Nome_Legivel', 'Métrica'),
        ('LUTs', 'LUTs'),
        ('Economia_LUTs (%)', 'Econ(%)'),
        ('SSIM_Mediana_Treino', 'Treino Md'),
        ('SSIM_Medio_Treino', 'Treino Média'),
        ('SSIM_Mediana_Teste', 'Teste Md'),
        ('SSIM_Medio_Teste', 'Teste Média'),
        ('PSNR_Teste (dB)', 'PSNR (dB)')
    ]
    sub = df_th[[orig for orig, _ in colunas if orig in df_th.columns]].copy()
    sub.columns = [novo for orig, novo in colunas if orig in df_th.columns]
    return sub.to_string(index=False)


def processar_threshold(
    th,
    iteracoes=50,
    alpha=0.5,
    k_vizinhos=10,
    dataset_treino=DATASET_TREINO_PADRAO,
    dataset_teste=DATASET_TESTE_PADRAO,
    pasta_saida="graficos_metricas",
    mostrar_graficos=False,
    verbose=True
):
    """Executa o benchmark e gera os gráficos de treino e teste para um threshold específico."""
    th_pct = int(round(th * 100))
    print("\n" + "=" * 78)
    print(f" INICIANDO BENCHMARK PARA THRESHOLD = {th_pct}% ({th})")
    print(f" Treino: '{dataset_treino}' | Teste: '{dataset_teste}' | Iterações: {iteracoes}")
    print("=" * 78)

    # 1. Executa o GRASP para as 3 métricas neste threshold (ssim_medio, ssim_mediana, ssim_mediageometrica)
    df_comp, dist_treino, dist_teste = executar_benchmark_metricas(
        thresholds=[th],
        metricas=[0, 1, 2],
        max_iteracoes=iteracoes,
        alpha=alpha,
        k_vizinhos=k_vizinhos,
        dataset_treino=dataset_treino,
        dataset_teste=dataset_teste,
        verbose=verbose,
        pasta_saida=pasta_saida,
        salvar_trace=True
    )

    if df_comp.empty:
        print(f"Nenhuma solução viável encontrada para TH = {th_pct}%.")
        return None

    # 2. Exibe tabela no terminal
    print("\n" + "-" * 78)
    print(f" RESULTADOS NUMÉRICOS: TREINO VS TESTE (TH = {th_pct}%)")
    print("-" * 78)
    print(formatar_tabela_resumo(df_comp))
    print("-" * 78)

    # 3. Salva CSV do threshold
    os.makedirs(pasta_saida, exist_ok=True)
    caminho_csv = os.path.join(pasta_saida, f"comparativo_metricas_th_{th_pct:02d}.csv")
    df_comp.to_csv(caminho_csv, index=False)
    print(f"  ✔ Dados salvos em CSV: '{caminho_csv}'")

    # 4. Gera os gráficos individuais de Treino e Teste + Comparativo
    prefixo = os.path.join(pasta_saida, "grafico_boxplot")
    fig_tr, fig_te, fig_comp = gerar_par_boxplots(
        dist_treino=dist_treino,
        dist_teste=dist_teste,
        th=th,
        prefixo=prefixo,
        salvar=True
    )

    # 5. Guia científico para auxílio na redação
    imprimir_guia_escolha_tcc(df_comp)

    # 6. Exibição interativa opcional
    if mostrar_graficos:
        plt.show()
    else:
        plt.close('all')

    return df_comp


def replotar_a_partir_do_csv(
    th,
    pasta_saida="graficos_metricas",
    dataset_treino=DATASET_TREINO_PADRAO,
    dataset_teste=DATASET_TESTE_PADRAO,
    mostrar_graficos=False
):
    """
    Gera instantaneamente os boxplots (Treino e Teste) a partir dos setups
    já salvos em 'comparativo_metricas_th_XX.csv', sem precisar reexecutar o GRASP.
    """
    th_pct = int(round(th * 100))
    caminho_csv = os.path.join(pasta_saida, f"comparativo_metricas_th_{th_pct:02d}.csv")
    if not os.path.exists(caminho_csv):
        caminho_alt = os.path.join(".", f"comparativo_metricas_th_{th_pct:02d}.csv")
        if os.path.exists(caminho_alt):
            caminho_csv = caminho_alt
        else:
            print(f"⚠️ Arquivo '{caminho_csv}' não encontrado para replotar.")
            return None

    os.makedirs(pasta_saida, exist_ok=True)

    import ast
    from grasp.avaliador import obter_valores_ssim
    df_comp = pd.read_csv(caminho_csv)
    # Filtra ssim_min caso ainda exista no CSV antigo
    df_comp = df_comp[df_comp['Metrica'] != 'ssim_min'].copy()
    caminho_saida_csv = os.path.join(pasta_saida, f"comparativo_metricas_th_{th_pct:02d}.csv")
    df_comp.to_csv(caminho_saida_csv, index=False)

    dist_treino = {}
    dist_teste = {}

    print(f"\n⚡ Replotando TH = {th_pct}% a partir de '{caminho_csv}' em segundos...")

    for _, row in df_comp.iterrows():
        m = row['Metrica']
        setup = row['Setup']
        if isinstance(setup, str):
            setup = ast.literal_eval(setup)

        dist_treino[(m, th)] = obter_valores_ssim(setup, dataset_file=dataset_treino)
        dist_teste[(m, th)] = obter_valores_ssim(setup, dataset_file=dataset_teste)

    prefixo = os.path.join(pasta_saida, "grafico_boxplot")
    fig_tr, fig_te, fig_comp = gerar_par_boxplots(
        dist_treino=dist_treino,
        dist_teste=dist_teste,
        th=th,
        prefixo=prefixo,
        salvar=True
    )

    # Re-gera o gráfico de convergência se o trace existir
    caminho_trace = os.path.join(pasta_saida, f"trace_grasp_th_{th_pct:02d}.csv")
    if not os.path.exists(caminho_trace):
        caminho_trace_alt = os.path.join(".", f"trace_grasp_th_{th_pct:02d}.csv")
        if os.path.exists(caminho_trace_alt):
            caminho_trace = caminho_trace_alt
    if os.path.exists(caminho_trace):
        from comparador_metricas import gerar_grafico_convergencia
        df_trace = pd.read_csv(caminho_trace)
        gerar_grafico_convergencia(df_trace, th, pasta_saida=pasta_saida, salvar=True)

    if mostrar_graficos:
        plt.show()
    else:
        plt.close('all')

    return df_comp


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark e Gerador de Boxplots (Treino vs Teste) das Métricas de Erro SSIM por Threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemplos:
  %(prog)s --th 0.15
  %(prog)s --th 0.10 0.15 0.20
  %(prog)s --th all
  %(prog)s --th 0.15 --iter 50 --show
        """
    )
    parser.add_argument(
        "--th", "--threshold",
        nargs="+",
        default=["0.15"],
        help="Threshold(s) de erro (ex: 0.15, ou 0.05 0.10 0.15, ou 'all' para todos os 6). Padrão: 0.15"
    )
    parser.add_argument(
        "--iter", "--iteracoes",
        type=int,
        default=50,
        help="Número de iterações do GRASP por busca (Padrão: 50)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Parâmetro alfa da RCL no GRASP (Padrão: 0.5)"
    )
    parser.add_argument(
        "--k-vizinhos",
        type=int,
        default=10,
        help="Quantidade de vizinhos na busca local do GRASP (Padrão: 10)"
    )
    parser.add_argument(
        "--dataset-treino",
        default=DATASET_TREINO_PADRAO,
        help=f"Arquivo com caminhos das imagens de treino (Padrão: {DATASET_TREINO_PADRAO})"
    )
    parser.add_argument(
        "--dataset-teste",
        default=DATASET_TESTE_PADRAO,
        help=f"Arquivo com caminhos das imagens de teste (Padrão: {DATASET_TESTE_PADRAO})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="graficos_metricas",
        help="Diretório onde salvar os gráficos (.png/.pdf) e arquivos CSV (Padrão: 'graficos_metricas')"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abre janelas gráficas interativas do matplotlib ao final"
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Gera instantaneamente os novos gráficos usando os setups já salvos nos arquivos CSV (sem reexecutar o GRASP)"
    )

    args = parser.parse_args()

    # Tratamento dos thresholds
    thresholds_selecionados = []
    for item in args.th:
        if str(item).lower() in ("all", "todos"):
            thresholds_selecionados = THRESHOLDS_PADRAO.copy()
            break
        try:
            # Suporta separação por vírgula se passado como "0.10,0.15"
            partes = str(item).split(",")
            for p in partes:
                val = float(p.strip())
                if val not in thresholds_selecionados:
                    thresholds_selecionados.append(val)
        except ValueError:
            print(f"⚠️ Valor de threshold inválido ignorado: '{item}'")

    if not thresholds_selecionados:
        thresholds_selecionados = [0.15]

    modo = "REPLOT INSTANTÂNEO (VIA CSVs EXISTENTES)" if args.replot else "BUSCA GRASP COMPLETA"
    print("\n" + "=" * 78)
    print(f"  BENCHMARK DE MÉTRICAS SSIM NO GRASP [{modo}]")
    print(f" Thresholds a processar: {thresholds_selecionados}")
    print(f" Pasta de saída: '{os.path.abspath(args.output_dir)}'")
    print("=" * 78)

    dfs = []
    for th in thresholds_selecionados:
        if args.replot:
            df = replotar_a_partir_do_csv(
                th=th,
                pasta_saida=args.output_dir,
                dataset_treino=args.dataset_treino,
                dataset_teste=args.dataset_teste,
                mostrar_graficos=args.show
            )
        else:
            df = processar_threshold(
                th=th,
                iteracoes=args.iter,
                alpha=args.alpha,
                k_vizinhos=args.k_vizinhos,
                dataset_treino=args.dataset_treino,
                dataset_teste=args.dataset_teste,
                pasta_saida=args.output_dir,
                mostrar_graficos=args.show,
                verbose=True
            )
        if df is not None:
            dfs.append(df)

    # Se processou mais de um threshold, salva também tabela consolidada geral
    if len(dfs) > 1:
        df_geral = pd.concat(dfs, ignore_index=True)
        caminho_geral = os.path.join(args.output_dir, "tabela_comparativa_metricas_consolidada.csv")
        df_geral.to_csv(caminho_geral, index=False)
        print("\n" + "=" * 78)
        print(f" ✔ Tabela consolidada com todos os thresholds salva em '{caminho_geral}'")
        print("=" * 78)

    print("\n Processamento concluído com sucesso!\n")


if __name__ == "__main__":
    main()
