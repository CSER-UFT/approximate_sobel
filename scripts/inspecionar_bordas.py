#!/usr/bin/env python3
"""
inspecionar_bordas.py
---------------------
Script para inspeção visual e comparativa dos resultados do Filtro Sobel
aproximado gerados pela exploração do espaço de projeto (DSE) com GRASP.

Permite conferir as imagens e métricas de qualidade (SSIM, Erro, MSE, PSNR)
sem depender da execução do GRASP ou do ambiente do Jupyter Notebook.

Exemplos de uso:
    python3 inspecionar_bordas.py --id 75
    python3 inspecionar_bordas.py --id 71,75,80
    python3 inspecionar_bordas.py --list
    python3 inspecionar_bordas.py --all --no-show
"""

import argparse
import ast
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJ_DIR not in sys.path:
    sys.path.insert(0, PROJ_DIR)

DATASET_DIR_PADRAO = os.path.join(PROJ_DIR, "image_dataset-main", "512x512")
CSV_RESULTADOS_PADRAO = (
    os.path.join(PROJ_DIR, "results", "resultados_dse_grasp.csv")
    if os.path.exists(os.path.join(PROJ_DIR, "results", "resultados_dse_grasp.csv"))
    else os.path.join(PROJ_DIR, "resultados_dse_grasp.csv")
)
TESTING_IN_PADRAO = os.path.join(PROJ_DIR, "testing.in")


def obter_custo_exato(df_resultados):
    """Recupera o custo em LUTs da arquitetura exata a partir do CSV."""
    df_exato = df_resultados[df_resultados["Limite_Erro"] == 0]
    if not df_exato.empty and "LUTs" in df_exato.columns:
        return int(df_exato["LUTs"].iloc[0])
    if "LUTs" in df_resultados.columns:
        return int(df_resultados["LUTs"].iloc[0])
    return 95


def encontrar_ou_gerar_exata(img_id, dataset_dir, img_orig_path, gerar_se_faltar=True):
    """Localiza ou gera a imagem exata de referência."""
    candidatos = [
        os.path.join(dataset_dir, f"{img_id}_precise.ppm"),
        os.path.join(dataset_dir, f"{img_id}.precise.sobel.ppm"),
    ]
    for c in candidatos:
        if os.path.isfile(c):
            return c

    if gerar_se_faltar and os.path.isfile(img_orig_path):
        try:
            from grasp.avaliador import executar_sobel
            print(f"⚙️ Imagem exata não encontrada para '{img_id}'. Gerando sob demanda via C++...")
            out = executar_sobel([0, 0, 0, 0, 0], img_orig_path, kind="precise")
            if out and os.path.isfile(out[0]):
                return out[0]
        except Exception as e:
            print(f"⚠️ Falha ao tentar gerar referência exata: {e}")

    return None


def encontrar_ou_gerar_aprox(img_id, dataset_dir, img_orig_path, row, gerar_se_faltar=True):
    """Localiza ou gera a imagem aproximada correspondente a um threshold."""
    th_val = float(row["Limite_Erro"])
    th_tag = int(round(th_val * 100))

    candidatos = [
        os.path.join(dataset_dir, f"{img_id}_th{th_tag:02d}.ppm"),
        os.path.join(dataset_dir, f"{img_id}.approx_th_{th_tag:02d}.sobel.ppm"),
    ]
    for c in candidatos:
        if os.path.isfile(c):
            return c

    if gerar_se_faltar and os.path.isfile(img_orig_path) and "Setup" in row:
        try:
            from grasp.avaliador import gerar_ppm_threshold
            setup_val = ast.literal_eval(str(row["Setup"])) if isinstance(row["Setup"], str) else row["Setup"]
            out = gerar_ppm_threshold(setup_val, th_val, dataset_file=img_orig_path)
            if out and os.path.isfile(out[0]):
                return out[0]
        except Exception as e:
            print(f"⚠️ Falha ao gerar aproximação para TH {th_tag}%: {e}")

    return None


def inspecionar_imagem(
    img_id,
    df_resultados,
    dataset_dir=DATASET_DIR_PADRAO,
    output_dir=".",
    mostrar=True,
    gerar_faltantes=True
):
    """Gera o painel comparativo visual e exibe as métricas de qualidade."""
    custo_exato = obter_custo_exato(df_resultados)
    df_aprox = df_resultados[df_resultados["Limite_Erro"] > 0].reset_index(drop=True)

    img_orig_path = os.path.join(dataset_dir, f"{img_id}.ppm")
    if not os.path.isfile(img_orig_path):
        print(f"❌ Imagem original '{img_orig_path}' não foi encontrada.")
        return False

    img_exata_path = encontrar_ou_gerar_exata(img_id, dataset_dir, img_orig_path, gerar_faltantes)
    if not img_exata_path or not os.path.isfile(img_exata_path):
        print(f"❌ Imagem de referência exata para '{img_id}' não encontrada.")
        return False

    img_orig = cv2.imread(img_orig_path, cv2.IMREAD_GRAYSCALE)
    img_exata = cv2.imread(img_exata_path, cv2.IMREAD_GRAYSCALE)

    total_colunas = len(df_aprox) + 2
    fig, axes = plt.subplots(1, total_colunas, figsize=(3.2 * total_colunas, 4.4))
    if total_colunas == 1:
        axes = [axes]

    # Painel 1: Original
    axes[0].imshow(img_orig, cmap="gray")
    axes[0].set_title(f"Original ({img_id}.ppm)\n(Input)\n---\n---\n---", fontsize=9, pad=8)
    axes[0].axis("off")

    # Painel 2: Exato
    axes[1].imshow(img_exata, cmap="gray")
    axes[1].set_title(
        f"Exato (Referência)\n({custo_exato} LUTs)\nErro: 0.0000\nSSIM: 1.0000\nPSNR: ∞ dB",
        fontsize=9,
        pad=8
    )
    axes[1].axis("off")

    # Tabela de métricas no terminal
    print("\n" + "=" * 70)
    print(f"  INSPEÇÃO VISUAL: Imagem {img_id}.ppm")
    print("=" * 70)
    print(f"{'Threshold':<12} | {'LUTs':<6} | {'SSIM':<8} | {'Erro':<8} | {'MSE':<8} | {'PSNR (dB)':<10}")
    print("-" * 70)
    print(f"{'0% (Exato)':<12} | {custo_exato:<6} | {'1.0000':<8} | {'0.0000':<8} | {'0.00':<8} | {'inf':<10}")

    for idx, row in df_aprox.iterrows():
        th_val = float(row["Limite_Erro"])
        th_tag = int(round(th_val * 100))
        img_aprox_path = encontrar_ou_gerar_aprox(img_id, dataset_dir, img_orig_path, row, gerar_faltantes)

        ax = axes[idx + 2]
        if img_aprox_path and os.path.isfile(img_aprox_path):
            img_aprox = cv2.imread(img_aprox_path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(img_aprox, cmap="gray")

            val_ssim = ssim(img_exata, img_aprox, data_range=255)
            val_erro = max(0.0, 1.0 - val_ssim)
            diff_sq = (img_exata.astype(np.float64) - img_aprox.astype(np.float64)) ** 2
            val_mse = float(np.mean(diff_sq))
            val_psnr = cv2.PSNR(img_exata, img_aprox) if val_mse > 0 else float("inf")

            ax.set_title(
                f"TH: {th_tag}% ({row['LUTs']} LUTs)\n"
                f"Erro: {val_erro:.4f}\n"
                f"SSIM: {val_ssim:.4f}\n"
                f"PSNR: {val_psnr:.2f} dB",
                fontsize=9,
                pad=8
            )
            print(f"{f'{th_tag}%':<12} | {row['LUTs']:<6} | {val_ssim:<8.4f} | {val_erro:<8.4f} | {val_mse:<8.2f} | {val_psnr:<10.2f}")
        else:
            ax.text(0.5, 0.5, "Não gerada", horizontalalignment="center", verticalalignment="center", fontsize=8)
            ax.set_title(f"TH: {th_tag}%\n({row['LUTs']} LUTs)", fontsize=9, pad=8)
            print(f"{f'{th_tag}%':<12} | {row['LUTs']:<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<10}")
        ax.axis("off")

    print("-" * 70)
    plt.suptitle(f"Comparação Visual do Filtro Sobel por Threshold ({os.path.basename(img_orig_path)})", fontsize=13, y=1.03)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_png_id = os.path.join(output_dir, f"comparacao_todos_thresholds_{img_id}.png")
    out_pdf_id = os.path.join(output_dir, f"comparacao_todos_thresholds_{img_id}.pdf")
    plt.savefig(out_png_id, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf_id, bbox_inches="tight")

    # Atualiza também a saída padrão
    out_png_padrao = os.path.join(output_dir, "comparacao_todos_thresholds.png")
    out_pdf_padrao = os.path.join(output_dir, "comparacao_todos_thresholds.pdf")
    plt.savefig(out_png_padrao, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf_padrao, bbox_inches="tight")

    print(f"✔ Gráficos salvos com sucesso:")
    print(f"   - {out_png_id}")
    print(f"   - {out_pdf_id}")

    if mostrar:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)

    return True


def listar_imagens_disponiveis(dataset_dir, testing_in):
    """Lista as imagens do dataset de teste disponíveis para inspeção."""
    print("📁 Imagens de Teste:")
    if os.path.isfile(testing_in):
        with open(testing_in, "r") as f:
            linhas = [line.strip() for line in f if line.strip() and not line.strip().isdigit()]
        ids = [os.path.splitext(os.path.basename(l))[0] for l in linhas]
        print(f"   Encontradas no '{testing_in}' ({len(ids)} imagens):")
        print("   " + ", ".join(ids[:20]) + ("..." if len(ids) > 20 else ""))
    else:
        ppm_files = [f for f in os.listdir(dataset_dir) if f.endswith(".ppm") and not ("sobel" in f or "precise" in f or "approx" in f or "th" in f)]
        ids = sorted([os.path.splitext(f)[0] for f in ppm_files])
        print(f"   Encontradas na pasta ({len(ids)} imagens):")
        print("   " + ", ".join(ids[:20]) + ("..." if len(ids) > 20 else ""))


def main():
    parser = argparse.ArgumentParser(
        description="Inspeção visual e cálculo de métricas das bordas Sobel geradas pelo DSE GRASP."
    )
    parser.add_argument(
        "--id", "-i",
        default="75",
        help="ID da imagem a ser inspecionada (ex: '75' ou múltiplos separados por vírgula '71,75,80'). Padrão: 75"
    )
    parser.add_argument(
        "--csv", "-c",
        default=CSV_RESULTADOS_PADRAO,
        help=f"Arquivo CSV com os resultados do DSE. Padrão: {CSV_RESULTADOS_PADRAO}"
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        default=DATASET_DIR_PADRAO,
        help=f"Pasta onde estão localizadas as imagens. Padrão: {DATASET_DIR_PADRAO}"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Diretório onde salvar as imagens geradas (.png e .pdf). Padrão: diretório atual"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Processa todas as imagens do dataset de teste listadas em testing.in."
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Lista as imagens disponíveis no dataset e sai."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Não abre a janela gráfica interativa do matplotlib (ideal para execução em lote ou terminal/SSH)."
    )
    parser.add_argument(
        "--sem-geracao",
        action="store_true",
        help="Não tenta gerar imagens aproximadas faltantes sob demanda."
    )

    args = parser.parse_args()

    if args.list:
        listar_imagens_disponiveis(args.dataset_dir, TESTING_IN_PADRAO)
        return

    if not os.path.isfile(args.csv):
        print(f"❌ Erro: Arquivo de resultados '{args.csv}' não encontrado!")
        print("   Certifique-se de executar o DSE ou apontar o caminho correto com '--csv <arquivo.csv>'.")
        sys.exit(1)

    df_resultados = pd.read_csv(args.csv)

    if args.all:
        if not os.path.isfile(TESTING_IN_PADRAO):
            print(f"❌ Arquivo '{TESTING_IN_PADRAO}' não encontrado para a opção --all.")
            sys.exit(1)
        with open(TESTING_IN_PADRAO, "r") as f:
            linhas = [line.strip() for line in f if line.strip() and not line.strip().isdigit()]
        ids_processar = [os.path.splitext(os.path.basename(l))[0] for l in linhas]
        args.no_show = True
    else:
        ids_processar = [i.strip() for i in args.id.split(",") if i.strip()]

    sucessos = 0
    for img_id in ids_processar:
        ok = inspecionar_imagem(
            img_id=img_id,
            df_resultados=df_resultados,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            mostrar=(not args.no_show and len(ids_processar) == 1),
            gerar_faltantes=(not args.sem_geracao)
        )
        if ok:
            sucessos += 1

    print(f"\n✨ Processamento concluído: {sucessos}/{len(ids_processar)} imagens inspecionadas com sucesso.")


if __name__ == "__main__":
    main()
