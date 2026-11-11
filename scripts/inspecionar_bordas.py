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
import math
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
    gerar_faltantes=True,
    colunas=4
):
    """Gera o painel comparativo visual em formato de grid e exibe as métricas de qualidade."""
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

    total_imagens = len(df_aprox) + 2
    ncols = min(max(1, int(colunas)), total_imagens)
    nrows = math.ceil(total_imagens / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.75 * ncols, 3.55 * nrows),
        sharex=True,
        sharey=True
    )
    axes_flat = np.array(axes).reshape(-1)

    def estilizar_eixo(ax, titulo_topo, texto_baixo=""):
        ax.set_title(titulo_topo, fontsize=13, fontweight="normal", pad=6)
        ax.tick_params(axis="both", which="major", labelsize=10)
        if texto_baixo:
            ax.set_xlabel(texto_baixo, fontsize=11.5, fontweight="normal", labelpad=6)

    # Painel 1: Original
    axes_flat[0].imshow(img_orig, cmap="gray")
    estilizar_eixo(axes_flat[0], "Original")

    # Painel 2: Exato
    axes_flat[1].imshow(img_exata, cmap="gray")
    estilizar_eixo(axes_flat[1], "Exato")

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

        ax = axes_flat[idx + 2]
        if img_aprox_path and os.path.isfile(img_aprox_path):
            img_aprox = cv2.imread(img_aprox_path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(img_aprox, cmap="gray")

            val_ssim = ssim(img_exata, img_aprox, data_range=255)
            val_erro = max(0.0, 1.0 - val_ssim)
            diff_sq = (img_exata.astype(np.float64) - img_aprox.astype(np.float64)) ** 2
            val_mse = float(np.mean(diff_sq))
            val_psnr = cv2.PSNR(img_exata, img_aprox) if val_mse > 0 else float("inf")

            estilizar_eixo(
                ax,
                f"Threshold: {th_tag}%",
                f"SSIM: {val_ssim:.4f}\nPSNR: {val_psnr:.2f}"
            )
            print(f"{f'{th_tag}%':<12} | {row['LUTs']:<6} | {val_ssim:<8.4f} | {val_erro:<8.4f} | {val_mse:<8.2f} | {val_psnr:<10.2f}")
        else:
            ax.text(0.5, 0.5, "Não gerada", horizontalalignment="center", verticalalignment="center", fontsize=11)
            estilizar_eixo(
                ax,
                f"Threshold: {th_tag}%",
                "SSIM: N/A\nPSNR: N/A"
            )
            print(f"{f'{th_tag}%':<12} | {row['LUTs']:<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<10}")

    # Esconde eixos extras se o grid não for múltiplo exato
    for extra_idx in range(total_imagens, len(axes_flat)):
        axes_flat[extra_idx].axis("off")

    print("-" * 70)
    plt.tight_layout(h_pad=2.4, w_pad=0.8)

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


def exportar_metricas_validacao_teste(
    df_resultados,
    dataset_dir=DATASET_DIR_PADRAO,
    testing_in=TESTING_IN_PADRAO,
    output_dir=".",
    gerar_faltantes=True
):
    """Calcula e exporta CSVs com SSIM, PSNR, MSE e Erro para todas as imagens de teste e todos os limiares."""
    if not os.path.isfile(testing_in):
        print(f"❌ Arquivo '{testing_in}' não encontrado.")
        return False

    with open(testing_in, "r") as f:
        linhas = [line.strip() for line in f if line.strip() and not line.strip().isdigit()]
    ids_teste = [os.path.splitext(os.path.basename(l))[0] for l in linhas]

    df_aprox = df_resultados[df_resultados["Limite_Erro"] > 0].reset_index(drop=True)
    registros_longos = []
    registros_largos = []

    print("\n" + "=" * 80)
    print(f"  VALIDAÇÃO COMPLETA DO CONJUNTO DE TESTE ({len(ids_teste)} imagens x {len(df_aprox)} thresholds)")
    print("=" * 80)

    for img_id in ids_teste:
        img_orig_path = os.path.join(dataset_dir, f"{img_id}.ppm")
        if not os.path.isfile(img_orig_path):
            print(f"⚠️ Imagem original '{img_orig_path}' não encontrada. Pulando...")
            continue

        img_exata_path = encontrar_ou_gerar_exata(img_id, dataset_dir, img_orig_path, gerar_faltantes)
        if not img_exata_path or not os.path.isfile(img_exata_path):
            print(f"⚠️ Imagem exata para '{img_id}' não encontrada. Pulando...")
            continue

        img_exata = cv2.imread(img_exata_path, cv2.IMREAD_GRAYSCALE)
        linha_larga = {"Imagem": f"{img_id}.ppm", "ID": int(img_id) if img_id.isdigit() else img_id}

        for _, row in df_aprox.iterrows():
            th_val = float(row["Limite_Erro"])
            th_tag = int(round(th_val * 100))
            img_aprox_path = encontrar_ou_gerar_aprox(img_id, dataset_dir, img_orig_path, row, gerar_faltantes)

            if img_aprox_path and os.path.isfile(img_aprox_path):
                img_aprox = cv2.imread(img_aprox_path, cv2.IMREAD_GRAYSCALE)
                val_ssim = float(ssim(img_exata, img_aprox, data_range=255))
                val_erro = max(0.0, 1.0 - val_ssim)
                diff_sq = (img_exata.astype(np.float64) - img_aprox.astype(np.float64)) ** 2
                val_mse = float(np.mean(diff_sq))
                val_psnr = float(cv2.PSNR(img_exata, img_aprox)) if val_mse > 0 else float("inf")
            else:
                val_ssim, val_erro, val_mse, val_psnr = np.nan, np.nan, np.nan, np.nan

            registros_longos.append({
                "ID": int(img_id) if img_id.isdigit() else img_id,
                "Imagem": f"{img_id}.ppm",
                "Threshold": f"{th_tag}%",
                "Limite_Erro": th_val,
                "LUTs": int(row["LUTs"]) if "LUTs" in row else np.nan,
                "Setup": row["Setup"] if "Setup" in row else "",
                "Somadores": row["Somadores"] if "Somadores" in row else "",
                "SSIM": round(val_ssim, 4) if not np.isnan(val_ssim) else np.nan,
                "Erro_SSIM": round(val_erro, 4) if not np.isnan(val_erro) else np.nan,
                "PSNR_dB": round(val_psnr, 2) if not np.isnan(val_psnr) else np.nan,
                "MSE": round(val_mse, 2) if not np.isnan(val_mse) else np.nan,
            })

            linha_larga[f"SSIM_{th_tag}%"] = round(val_ssim, 4)
            linha_larga[f"PSNR_{th_tag}%"] = round(val_psnr, 2)
            linha_larga[f"MSE_{th_tag}%"] = round(val_mse, 2)

        registros_largos.append(linha_larga)

    df_longo = pd.DataFrame(registros_longos)
    df_largo = pd.DataFrame(registros_largos)

    # Adiciona estatísticas de resumo ao final da tabela larga
    colunas_metricas = [c for c in df_largo.columns if c not in ("Imagem", "ID")]
    resumo_rows = []
    for nome_est, func in [
        ("Média Aritmética", lambda s: np.nanmean(s)),
        ("Média Geométrica", lambda s: float(np.exp(np.nanmean(np.log(np.clip(s, 1e-12, None)))))),
        ("Desvio Padrão", lambda s: np.nanstd(s, ddof=1)),
        ("Mínimo", lambda s: np.nanmin(s)),
        ("Máximo", lambda s: np.nanmax(s)),
    ]:
        r = {"Imagem": nome_est, "ID": "-"}
        for col in colunas_metricas:
            casas = 4 if col.startswith("SSIM_") else 2
            r[col] = round(float(func(df_largo[col].values.astype(float))), casas)
        resumo_rows.append(r)

    df_largo_com_resumo = pd.concat([df_largo, pd.DataFrame(resumo_rows)], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    csv_detalhado = os.path.join(output_dir, "validacao_teste_detalhado.csv")
    csv_por_imagem = os.path.join(output_dir, "validacao_teste_por_imagem.csv")

    df_longo.to_csv(csv_detalhado, index=False)
    df_largo_com_resumo.to_csv(csv_por_imagem, index=False)

    print(df_largo_com_resumo.to_string(index=False))
    print("-" * 80)
    print("✔ Arquivos CSV de validação do teste salvos com sucesso:")
    print(f"   1. Formato detalhado (180 linhas): {csv_detalhado}")
    print(f"   2. Formato tabela por imagem:      {csv_por_imagem}")
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
    output_dir_padrao = os.path.join(PROJ_DIR, "results") if os.path.isdir(os.path.join(PROJ_DIR, "results")) else "."
    parser.add_argument(
        "--output-dir", "-o",
        default=output_dir_padrao,
        help=f"Diretório onde salvar as imagens geradas (.png e .pdf). Padrão: {output_dir_padrao}"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Número de colunas no grid visual (ex: 4 para grid 2x4 com 8 imagens). Padrão: 4"
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
        "--tabela-teste", "-t",
        action="store_true",
        help="Calcula e exporta CSVs com as métricas de validação (SSIM, PSNR, MSE) de todas as imagens de teste em todos os limiares."
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

    if args.tabela_teste:
        exportar_metricas_validacao_teste(
            df_resultados=df_resultados,
            dataset_dir=args.dataset_dir,
            testing_in=TESTING_IN_PADRAO,
            output_dir=args.output_dir,
            gerar_faltantes=(not args.sem_geracao)
        )
        if not args.all:
            return

    if args.all:
        exportar_metricas_validacao_teste(
            df_resultados=df_resultados,
            dataset_dir=args.dataset_dir,
            testing_in=TESTING_IN_PADRAO,
            output_dir=args.output_dir,
            gerar_faltantes=(not args.sem_geracao)
        )
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
            gerar_faltantes=(not args.sem_geracao),
            colunas=args.cols
        )
        if ok:
            sucessos += 1

    print(f"\n✨ Processamento concluído: {sucessos}/{len(ids_processar)} imagens inspecionadas com sucesso.")


if __name__ == "__main__":
    main()
