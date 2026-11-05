#!/usr/bin/env python3
"""
comparar_grasp_vs_aleatorio.py

Gera o gráfico e relatório comparativo entre as soluções encontradas pelo GRASP
e as 100 configurações aleatórias, com tratamento de sobreposição de textos.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COR_GRASP = "#0017e6"       
COR_TEXTO = "#0d0d0e"

def extrair_metricas_xml(xml_path):
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        res = root.find(".//AreaEstimates/Resources")
        luts = int(res.find("LUT").text.strip()) if res is not None and res.find("LUT") is not None else None
        ffs = int(res.find("FF").text.strip()) if res is not None and res.find("FF") is not None else None
        est_clk = float(root.find(".//EstimatedClockPeriod").text.strip()) if root.find(".//EstimatedClockPeriod") is not None else None
        fmax_mhz = round(1000.0 / est_clk, 2) if est_clk and est_clk > 0 else None
        return {"LUT_full": luts, "FF_full": ffs, "Fmax_MHz": fmax_mhz}
    except Exception:
        return {}


PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_DIR = os.path.join(PROJ_DIR, "results")

def _default_path(filename):
    cand = os.path.join(RESULTS_DIR, filename)
    return cand if os.path.exists(cand) else os.path.join(PROJ_DIR, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Gera gráfico comparativo entre o GRASP e as 100 configurações aleatórias."
    )
    parser.add_argument(
        "--csv-aleatorio",
        default=_default_path("configuracoes_aleatorias_100.csv"),
        help="Arquivo CSV das 100 configurações aleatórias"
    )
    parser.add_argument(
        "--csv-dse",
        default=_default_path("resultados_dse_grasp.csv"),
        help="Arquivo CSV do DSE GRASP"
    )
    parser.add_argument(
        "--pasta-reports",
        default=_default_path("reports_sintese_aleatorio"),
        help="Pasta de relatórios Vitis HLS"
    )
    parser.add_argument(
        "--saida",
        default=os.path.join(RESULTS_DIR, "grafico_grasp_vs_amostras_aleatorias.png"),
        help="Nome da imagem de saída"
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv_aleatorio):
        print(f"Aviso: Arquivo '{args.csv_aleatorio}' não encontrado. O gráfico será gerado apenas com os dados do GRASP.")
        has_rand = False
    else:
        has_rand = True

    if not os.path.exists(args.csv_dse):
        print(f"Erro: Arquivo DSE '{args.csv_dse}' não encontrado!")
        sys.exit(1)

    df_grasp = pd.read_csv(args.csv_dse)
    df_rand = pd.read_csv(args.csv_aleatorio) if has_rand else None

    print("=" * 78)
    print(" COMPARATIVO: GRASP vs 100 AMOSTRAS ALEATÓRIAS")
    print(f" Arquivo GRASP DSE: '{args.csv_dse}' ({len(df_grasp)} configurações)")
    if has_rand:
        print(f" Arquivo Aleatório: '{args.csv_aleatorio}' ({len(df_rand)} configurações)")
    print("=" * 78)

    # Processamento das amostras aleatórias (se existirem)
    if has_rand:
        relatorios_encontrados = 0
        if os.path.isdir(args.pasta_reports):
            luts_full = []
            for _, row in df_rand.iterrows():
                tag = row["ID"]
                xml_p = os.path.join(args.pasta_reports, f"{tag}_csynth.xml")
                if os.path.exists(xml_p):
                    m = extrair_metricas_xml(xml_p)
                    luts_full.append(m.get("LUT_full", np.nan))
                    relatorios_encontrados += 1
                else:
                    luts_full.append(np.nan)
            df_rand["LUTs_Full_Design"] = luts_full

        df_amostras = df_rand[df_rand["ID"] != "rand_000"].copy()
        df_amostras["SSIM_Teste_Mediana"] = pd.to_numeric(df_amostras["SSIM_Teste_Mediana"], errors="coerce")
        df_amostras["LUTs_Estimadas"] = pd.to_numeric(df_amostras["LUTs_Estimadas"], errors="coerce")
        df_amostras["Erro_Teste"] = 1.0 - df_amostras["SSIM_Teste_Mediana"]
    else:
        df_amostras = None

    # Processamento do GRASP usando explicitamente o Erro_Teste real do CSV
    df_grasp["SSIM_Teste"] = pd.to_numeric(df_grasp["SSIM_Teste"], errors="coerce")
    df_grasp["LUTs"] = pd.to_numeric(df_grasp["LUTs"], errors="coerce")
    df_grasp["Erro_Teste"] = pd.to_numeric(df_grasp["Erro_Teste"], errors="coerce")

    # Ordena e remove duplicatas exatas de erro e LUTs (ex: conflito entre 25% e 30% com o mesmo resultado)
    df_grasp_sorted = (
        df_grasp.sort_values(by="Erro_Teste", kind="stable")
        .drop_duplicates(subset=["Erro_Teste", "LUTs"], keep="first")
    )

    # Criação da Figura
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # 1. Plota as amostras aleatórias (nuvem de dispersão)
    if df_amostras is not None:
        ax.scatter(
            df_amostras["Erro_Teste"],
            df_amostras["LUTs_Estimadas"],
            color="#7f8c8d",
            alpha=0.45,
            s=45,
            edgecolors="none",
            label="100 Amostras Aleatórias"
        )

    # 2. Plota a Fronteira de Pareto do GRASP
    ax.plot(
        df_grasp_sorted["Erro_Teste"],
        df_grasp_sorted["LUTs"],
        color=COR_GRASP,
        linestyle="-",
        linewidth=2.4,
        zorder=4,
        label="Fronteira de Soluções (GRASP)"
    )
    ax.scatter(
        df_grasp_sorted["Erro_Teste"],
        df_grasp_sorted["LUTs"],
        color=COR_GRASP,
        s=95,
        marker="o",
        edgecolors="#ffffff",
        linewidth=1.5,
        zorder=5,
        label="Pontos Selecionados pelo GRASP"
    )

    # 3. Caixinhas com posicionamento alternado (acima/abaixo) para evitar sobreposição
    for i, (_, row) in enumerate(df_grasp_sorted.iterrows()):
        erro_val = row["Erro_Teste"]
        luts_val = row["LUTs"]
        eh_exato = row["Limite_Erro"] == 0

        texto = "Exato" if eh_exato else f"Err: {erro_val:.3f}"

        # Alterna a posição vertical da caixa: índice par abaixo (-22), índice ímpar acima (+18)
        offset_y = -24 if i % 2 == 0 else 18

        ax.annotate(
            texto,
            (erro_val, luts_val),
            textcoords="offset points",
            xytext=(0, offset_y),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=COR_TEXTO,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COR_GRASP, alpha=0.88, lw=0.7)
        )

    ax.set_title(
        "Algoritmo GRASP vs. 100 Amostras (Fronteira de Pareto)\n",
        fontsize=12,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel("Erro Real no Conjunto de Teste (1 - SSIM)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Consumo de LUTs", fontsize=11, fontweight="bold", labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)

    # Ajuste de limites com margem de segurança maior para as caixas superiores e inferiores
    max_err = df_amostras["Erro_Teste"].max() if df_amostras is not None else df_grasp["Erro_Teste"].max()
    ax.set_xlim(-0.02, max(max_err + 0.05, 0.35))
    min_luts = min(df_amostras["LUTs_Estimadas"].min() if df_amostras is not None else 95, df_grasp["LUTs"].min())
    ax.set_ylim(min_luts - 15, 105)

    plt.tight_layout()

    # Salva em PNG e PDF
    nome_pdf = args.saida.replace(".png", ".pdf") if args.saida.endswith(".png") else f"{args.saida}.pdf"
    plt.savefig(args.saida, dpi=300, bbox_inches="tight")
    plt.savefig(nome_pdf, bbox_inches="tight")
    print(f"\n Gráfico gerado e salvo com sucesso:")

if __name__ == "__main__":
    main()