"""
comparador_metricas.py

Módulo para benchmark e geração de gráficos comparativos entre as diferentes
métricas de erro SSIM (ssim_medio, ssim_mediana, ssim_mediageometrica, ssim_min)
para embasar decisões de projeto e fundamentação teórica de TCC/artigos científicos.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Garante importação do pacote grasp na raiz do projeto
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJ_DIR not in sys.path:
    sys.path.insert(0, PROJ_DIR)

from grasp.main_grasp import grasp_sobel, zerar_contador_combinacoes_grasp
from grasp.avaliador import (
    avaliar_no_teste,
    avaliar_no_treino,
    obter_valores_ssim,
    nomes_ssim,
    obter_nome_ssim,
    DATASET_TREINO_PADRAO,
    DATASET_TESTE_PADRAO
)

# Paleta de cores padronizada e profissional para publicações
CORES_METRICAS = {
    'ssim_medio': '#1f77b4',           # Azul clássico
    'ssim_mediana': '#2ca02c',         # Verde
    'ssim_mediageometrica': '#ff7f0e', # Laranja
    'ssim_min': '#d62728'              # Vermelho (pior caso / rigoroso)
}

NOMES_LEGIVEIS = {
    'ssim_medio': 'SSIM Médio (Média Aritmética)',
    'ssim_mediana': 'SSIM Mediana (Robusta a Outliers)',
    'ssim_mediageometrica': 'SSIM Geométrico (Média Geométrica)',
    'ssim_min': 'SSIM Mínimo (Pior Caso Garantido)'
}

ROTULOS_EXIBICAO = {
    'ssim_medio': 'SSIM Médio\n(Aritmética)',
    'ssim_mediana': 'SSIM Mediana\n(Robusta)',
    'ssim_mediageometrica': 'SSIM Geométrico\n(Geométrica)',
    'ssim_min': 'SSIM Mínimo\n(Pior Caso)'
}


def executar_benchmark_metricas(
    thresholds=None,
    metricas=None,
    max_iteracoes=50,
    alpha=0.5,
    k_vizinhos=10,
    dataset_treino=DATASET_TREINO_PADRAO,
    dataset_teste=DATASET_TESTE_PADRAO,
    verbose=True,
    pasta_saida="graficos_metricas",
    salvar_trace=True
):
    """
    Executa o GRASP para cada uma das métricas SSIM e para cada threshold,
    coletando dados de área (LUTs), métricas no treino e no teste, distribuições
    individuais de SSIM para ambos os conjuntos, e trace detalhado por iteração.

    Retorna:
      df_comparativo: pd.DataFrame com as métricas consolidadas (treino e teste).
      distribuicoes_treino: dict (nome_metrica, th) -> list[float] de SSIMs no Treino.
      distribuicoes_teste: dict (nome_metrica, th) -> list[float] de SSIMs no Teste.
    """
    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)]
    elif thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    if isinstance(metricas, (int, str)):
        metricas = [metricas]
    elif metricas is None:
        metricas = [0, 1, 2]

    registros = []
    distribuicoes_treino = {}
    distribuicoes_teste = {}

    if verbose:
        print("=" * 75)
        print(" INICIANDO BENCHMARK COMPARATIVO DAS MÉTRICAS DE ERRO SSIM")
        print(f" Treino: {dataset_treino} | Teste: {dataset_teste}")
        print(f" Thresholds: {thresholds} | Iterações GRASP: {max_iteracoes}")
        print("=" * 75)

    for th in thresholds:
        th_pct = int(round(th * 100))
        traces_threshold = []
        if verbose:
            print(f"\n======================================================================")
            print(f" >>> OTIMIZANDO THRESHOLD {th_pct}% ({th}) PARA AS {len(metricas)} MÉTRICAS")
            print(f"======================================================================")

        for m in metricas:
            nome_m = obter_nome_ssim(m)
            rotulo_legivel = NOMES_LEGIVEIS.get(nome_m, nome_m)
            if verbose:
                print(f"\n  ▶ Buscando com métrica: [{nome_m}] - {rotulo_legivel}...")

            setup, area, respeitou_limite, trace_m = grasp_sobel(
                configuracao_exata=[0, 0, 0, 0, 0],
                limite_erro=th,
                max_iteracoes=max_iteracoes,
                alpha=alpha,
                total_somadores_bib=11,
                total_somadores_app=5,
                k_vizinhos=k_vizinhos,
                retornar_combinacoes=False,
                retornar_trace=True,
                dataset_file=dataset_treino,
                funcao_erro=m
            )

            # Enriquece cada registro do trace com os metadados do experimento
            for item in trace_m:
                row_copy = dict(item)
                row_copy['Threshold'] = th
                row_copy['Threshold_Tag'] = f"{th_pct}%"
                row_copy['Metrica'] = nome_m
                row_copy['Nome_Legivel'] = rotulo_legivel
                traces_threshold.append(row_copy)

            if setup and respeitou_limite:
                # 1. Avaliação completa no conjunto de treino (70 imagens)
                erro_tr, ssim_tr, mse_tr, psnr_tr = avaliar_no_treino(
                    setup, dataset_file=dataset_treino, funcao_erro=m
                )
                ssims_treino = obter_valores_ssim(setup, dataset_file=dataset_treino)
                distribuicoes_treino[(nome_m, th)] = ssims_treino

                # 2. Avaliação completa no conjunto de teste independente (30 imagens)
                erro_t, ssim_t, mse_t, psnr_t = avaliar_no_teste(
                    setup, dataset_file=dataset_teste, funcao_erro=m
                )
                ssims_teste = obter_valores_ssim(setup, dataset_file=dataset_teste)
                distribuicoes_teste[(nome_m, th)] = ssims_teste

                economia = (1.0 - (area / 95.0)) * 100.0

                registros.append({
                    'Threshold': th,
                    'Threshold_Tag': f"{th_pct}%",
                    'Metrica': nome_m,
                    'Nome_Legivel': rotulo_legivel,
                    'Setup': str(setup),
                    'LUTs': area,
                    'Economia_LUTs (%)': round(economia, 1),
                    # Treino
                    'SSIM_Medio_Treino': round(float(np.mean(ssims_treino)), 4),
                    'SSIM_Mediana_Treino': round(float(np.median(ssims_treino)), 4),
                    'SSIM_Min_Treino': round(float(np.min(ssims_treino)), 4),
                    'Desvio_Treino': round(float(np.std(ssims_treino)), 4),
                    'PSNR_Treino (dB)': round(psnr_tr, 2) if psnr_tr != float('inf') else 99.9,
                    'MSE_Treino': round(mse_tr, 2),
                    # Teste
                    'SSIM_Medio_Teste': round(float(np.mean(ssims_teste)), 4),
                    'SSIM_Mediana_Teste': round(float(np.median(ssims_teste)), 4),
                    'SSIM_Min_Teste': round(float(np.min(ssims_teste)), 4),
                    'Desvio_Teste': round(float(np.std(ssims_teste)), 4),
                    'PSNR_Teste (dB)': round(psnr_t, 2) if psnr_t != float('inf') else 99.9,
                    'MSE_Teste': round(mse_t, 2)
                })

                if verbose:
                    print(f"    ✓ [{nome_m}] Área: {area} LUTs ({economia:.1f}%) | Treino Md: {np.median(ssims_treino):.4f} | Teste Md: {np.median(ssims_teste):.4f}")
            else:
                if verbose:
                    print(f"    ⚠️ [{nome_m}] Nenhuma solução viável para o threshold {th_pct}%.")

        # Salva o arquivo de trace e gera o gráfico de convergência para o threshold
        if salvar_trace and traces_threshold and pasta_saida:
            os.makedirs(pasta_saida, exist_ok=True)
            caminho_trace = os.path.join(pasta_saida, f"trace_grasp_th_{th_pct:02d}.csv")
            df_trace = pd.DataFrame(traces_threshold)
            df_trace.to_csv(caminho_trace, index=False)
            if verbose:
                print(f"\n  ✔ Trace por iteração salvo em: '{caminho_trace}'")
            gerar_grafico_convergencia(df_trace, th, pasta_saida=pasta_saida)

    df_comp = pd.DataFrame(registros)
    return df_comp, distribuicoes_treino, distribuicoes_teste


def gerar_grafico_boxplot_isolado(
    distribuicoes,
    th,
    tipo_dataset="Teste",
    df_comp=None,
    salvar_arquivo=None,
    limites_y=None
):
    """
    Gera um gráfico Boxplot isolado para um único threshold específico
    (4 caixas destacadas: Média, Mediana, Geométrica, Mínimo).
    """
    th_pct = int(round(th * 100))
    metricas_presentes = [m for m in nomes_ssim if (m, th) in distribuicoes]
    if not metricas_presentes:
        metricas_presentes = [m for (m, th_cand) in distribuicoes.keys() if th_cand == th]

    if not metricas_presentes:
        print(f"❌ Nenhuma distribuição encontrada para threshold {th_pct}%.")
        return None

    dados_box = [distribuicoes[(m, th)] for m in metricas_presentes]
    labels_box = [ROTULOS_EXIBICAO.get(m, m.replace("ssim_", "")) for m in metricas_presentes]
    n_amostras = len(dados_box[0]) if dados_box and len(dados_box[0]) > 0 else 0

    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=100)

    try:
        bp = ax.boxplot(
            dados_box,
            tick_labels=labels_box,
            patch_artist=True,
            widths=0.52,
            showmeans=True,
            meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': '#222222', 'markersize': 7},
            medianprops={'color': '#111111', 'linewidth': 2.2},
            whiskerprops={'color': '#444444', 'linewidth': 1.4, 'linestyle': '--'},
            capprops={'color': '#444444', 'linewidth': 1.5},
            flierprops={'marker': 'o', 'markerfacecolor': '#e74c3c', 'markeredgecolor': 'none', 'markersize': 5, 'alpha': 0.65}
        )
    except TypeError:
        bp = ax.boxplot(
            dados_box,
            labels=labels_box,
            patch_artist=True,
            widths=0.52,
            showmeans=True,
            meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': '#222222', 'markersize': 7},
            medianprops={'color': '#111111', 'linewidth': 2.2},
            whiskerprops={'color': '#444444', 'linewidth': 1.4, 'linestyle': '--'},
            capprops={'color': '#444444', 'linewidth': 1.5},
            flierprops={'marker': 'o', 'markerfacecolor': '#e74c3c', 'markeredgecolor': 'none', 'markersize': 5, 'alpha': 0.65}
        )

    for patch, m in zip(bp['boxes'], metricas_presentes):
        patch.set_facecolor(CORES_METRICAS.get(m, '#4a90e2'))
        patch.set_alpha(0.75)
        patch.set_edgecolor('#222222')
        patch.set_linewidth(1.3)

    for idx, d in enumerate(dados_box, start=1):
        med_val = float(np.median(d))
        mean_val = float(np.mean(d))
        ax.annotate(
            f"Md: {med_val:.3f}\nMédia: {mean_val:.3f}",
            (idx, med_val),
            textcoords="offset points",
            xytext=(28, 0),
            va='center',
            fontsize=8.5,
            fontweight='bold',
            color='#111111',
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.9, lw=0.8)
        )

    ax.set_title(
        f"Dispersão de SSIM no {tipo_dataset.upper()} por Métrica (Threshold = {th_pct}%)\n"
        f"[Avaliação sobre {n_amostras} Imagens]",
        fontsize=12,
        fontweight='bold',
        pad=14,
        ha='center'
    )
    ax.set_xlabel("Função de Agregação de Erro no GRASP", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel(f"Índice de Similaridade Estrutural (SSIM no {tipo_dataset})", fontsize=11, fontweight='bold', labelpad=10)

    if limites_y:
        ax.set_ylim(limites_y)

    ax.yaxis.grid(True, linestyle="--", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0.4, len(dados_box) + 0.7)

    elementos_legenda = [
        Line2D([0], [0], color='#111111', lw=2.2, label='Mediana (Md)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='#222', markersize=7, label='Média'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=6, label='Outlier')
    ]
    ax.legend(handles=elementos_legenda, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()

    if salvar_arquivo:
        plt.savefig(salvar_arquivo, dpi=300, bbox_inches='tight')
        nome_pdf = salvar_arquivo.replace(".png", ".pdf") if salvar_arquivo.endswith(".png") else f"{salvar_arquivo}.pdf"
        plt.savefig(nome_pdf, bbox_inches='tight')
        print(f"  ✔ Salvo: '{salvar_arquivo}' e '{nome_pdf}'")

    return fig


def gerar_grafico_comparativo_lado_a_lado(
    dist_treino,
    dist_teste,
    th,
    salvar_arquivo=None,
    limites_y=None
):
    """
    Gera uma figura de 2 painéis lado a lado (Esquerda: Treino, Direita: Teste)
    compartilhando o mesmo eixo Y para facilitar a comparação visual no TCC.
    """
    th_pct = int(round(th * 100))
    metricas_presentes = [m for m in nomes_ssim if (m, th) in dist_treino and (m, th) in dist_teste]
    if not metricas_presentes:
        return None

    ROTULOS = {
        'ssim_medio': 'SSIM Médio',
        'ssim_mediana': 'SSIM Mediana',
        'ssim_mediageometrica': 'SSIM Geométrico',
        'ssim_min': 'SSIM Mínimo'
    }

    dados_tr = [dist_treino[(m, th)] for m in metricas_presentes]
    dados_te = [dist_teste[(m, th)] for m in metricas_presentes]
    labels = [ROTULOS.get(m, m) for m in metricas_presentes]

    n_tr = len(dados_tr[0]) if dados_tr else 70
    n_te = len(dados_te[0]) if dados_te else 30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, dpi=100)

    for ax, dados, titulo, n_imgs in [
        (ax1, dados_tr, f"(A) Conjunto de Treino ({n_tr} Imagens)", n_tr),
        (ax2, dados_te, f"(B) Conjunto de Teste ({n_te} Imagens Inéditas)", n_te)
    ]:
        try:
            bp = ax.boxplot(
                dados,
                tick_labels=labels,
                patch_artist=True,
                widths=0.50,
                showmeans=True,
                meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': '#222222', 'markersize': 6.5},
                medianprops={'color': '#111111', 'linewidth': 2.0},
                whiskerprops={'color': '#444444', 'linewidth': 1.3, 'linestyle': '--'},
                capprops={'color': '#444444', 'linewidth': 1.4},
                flierprops={'marker': 'o', 'markerfacecolor': '#e74c3c', 'markeredgecolor': 'none', 'markersize': 4.5, 'alpha': 0.6}
            )
        except TypeError:
            bp = ax.boxplot(
                dados,
                labels=labels,
                patch_artist=True,
                widths=0.50,
                showmeans=True,
                meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': '#222222', 'markersize': 6.5},
                medianprops={'color': '#111111', 'linewidth': 2.0},
                whiskerprops={'color': '#444444', 'linewidth': 1.3, 'linestyle': '--'},
                capprops={'color': '#444444', 'linewidth': 1.4},
                flierprops={'marker': 'o', 'markerfacecolor': '#e74c3c', 'markeredgecolor': 'none', 'markersize': 4.5, 'alpha': 0.6}
            )

        for patch, m in zip(bp['boxes'], metricas_presentes):
            patch.set_facecolor(CORES_METRICAS.get(m, '#4a90e2'))
            patch.set_alpha(0.75)
            patch.set_edgecolor('#222222')
            patch.set_linewidth(1.2)

        for idx, d in enumerate(dados, start=1):
            med_val = float(np.median(d))
            mean_val = float(np.mean(d))
            ax.annotate(
                f"Md: {med_val:.3f}\nMédia: {mean_val:.3f}",
                (idx, med_val),
                textcoords="offset points",
                xytext=(24, 0),
                va='center',
                fontsize=8.0,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.85, lw=0.8)
            )

        ax.set_title(titulo, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Métrica de Erro no GRASP", fontsize=10, fontweight='bold', labelpad=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.55, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0.4, len(dados) + 0.7)

    ax1.set_ylabel("Índice de Similaridade Estrutural (SSIM)", fontsize=11, fontweight='bold', labelpad=8)

    elementos_legenda = [
        Line2D([0], [0], color='#111111', lw=2.0, label='Mediana (Md)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='#222', markersize=6.5, label='Média'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=4.5, label='Outlier')
    ]
    ax2.legend(handles=elementos_legenda, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.9, fontsize=8.5)

    if limites_y:
        ax1.set_ylim(limites_y)

    fig.suptitle(f"Comparativo de Generalização: Treino vs. Teste (Threshold = {th_pct}%)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if salvar_arquivo:
        plt.savefig(salvar_arquivo, dpi=300, bbox_inches='tight')
        nome_pdf = salvar_arquivo.replace(".png", ".pdf") if salvar_arquivo.endswith(".png") else f"{salvar_arquivo}.pdf"
        plt.savefig(nome_pdf, bbox_inches='tight')
        print(f"  ✔ Salvo Comparativo Lado a Lado: '{salvar_arquivo}' e '{nome_pdf}'")

    return fig


def gerar_par_boxplots(dist_treino, dist_teste, th, prefixo="grafico_boxplot", salvar=True):
    """
    Gera os 2 gráficos individuais (1 de Treino e 1 de Teste) para o threshold especificado,
    e também gera um gráfico comparativo lado a lado.
    """
    th_pct = int(round(th * 100))

    arq_tr_png = f"{prefixo}_treino_th_{th_pct:02d}.png" if salvar else None
    arq_te_png = f"{prefixo}_teste_th_{th_pct:02d}.png" if salvar else None

    # Determina limites comuns de Y para os dois gráficos
    vals_tr = [val for (m, t), arr in dist_treino.items() if t == th for val in arr]
    vals_te = [val for (m, t), arr in dist_teste.items() if t == th for val in arr]
    todos_vals = vals_tr + vals_te
    lim_y = (max(0.0, min(todos_vals) - 0.04), min(1.02, max(todos_vals) + 0.03)) if todos_vals else None

    # 1. Gráfico de Treino
    fig_tr = gerar_grafico_boxplot_isolado(
        dist_treino, th, tipo_dataset="Treino", salvar_arquivo=arq_tr_png, limites_y=lim_y
    )

    # 2. Gráfico de Teste
    fig_te = gerar_grafico_boxplot_isolado(
        dist_teste, th, tipo_dataset="Teste", salvar_arquivo=arq_te_png, limites_y=lim_y
    )

    # 3. Gráfico Comparativo lado a lado
    arq_comp_png = f"{prefixo}_comparativo_th_{th_pct:02d}.png" if salvar else None
    fig_comp = gerar_grafico_comparativo_lado_a_lado(
        dist_treino, dist_teste, th, salvar_arquivo=arq_comp_png, limites_y=lim_y
    )

    return fig_tr, fig_te, fig_comp


def gerar_grafico_convergencia(df_trace, th, pasta_saida="graficos_metricas", salvar=True):
    """
    Gera o gráfico de convergência da meta-heurística GRASP (Evolução da Melhor Área por Iteração)
    para o threshold especificado, comparando as métricas avaliadas.
    Salva em PNG (300 DPI) e PDF vetorial na pasta de saída com a legenda externa.
    """
    if df_trace is None or df_trace.empty:
        return None

    th_pct = int(round(th * 100))
    metricas_presentes = [m for m in nomes_ssim if m in df_trace['Metrica'].unique()]
    if not metricas_presentes:
        return None

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=100)

    # Linha base do filtro Sobel exato (todas as 5 posições com somador exato = 95 LUTs)
    ax.axhline(95, color='#7f8c8d', linestyle=':', linewidth=1.5, label='Baseline Exato (95 LUTs)', zorder=1)

    for m in metricas_presentes:
        sub = df_trace[df_trace['Metrica'] == m].copy()
        if sub.empty:
            continue
        cor = CORES_METRICAS.get(m, '#1f77b4')
        nome = NOMES_LEGIVEIS.get(m, m)

        # Trata melhor área viável: preenche para frente os valores
        sub['Melhor_Area_Global'] = sub['Melhor_Area_Global'].ffill()

        # Linha contínua da melhor solução viável ao longo das iterações
        ax.plot(
            sub['Iteracao'],
            sub['Melhor_Area_Global'],
            color=cor,
            linewidth=2.2,
            label=nome,
            zorder=3
        )

        # Destaca as iterações em que houve novo recorde viável
        recordes = sub[sub['Bateu_Recorde'] == True]
        if not recordes.empty:
            ax.scatter(
                recordes['Iteracao'],
                recordes['Melhor_Area_Global'],
                color=cor,
                edgecolors='#111111',
                s=65,
                linewidth=1.2,
                zorder=4
            )

    max_iter = int(df_trace['Iteracao'].max()) if not df_trace.empty else 50
    step_x = max(1, max_iter // 10)
    ax.set_xticks(list(range(1, max_iter + 1, step_x)) + ([max_iter] if max_iter % step_x != 1 else []))
    ax.set_xlim(0.5, max_iter + 0.5)

    ax.set_title(
        f"Convergência do GRASP por Iteração (Threshold = {th_pct}%)\n"
        f"[Evolução da Redução de Área em LUTs ao Longo da Otimização]",
        fontsize=12,
        fontweight='bold',
        pad=14,
        ha='center'
    )
    ax.set_xlabel("Iteração do GRASP", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Melhor Área Encontrada (LUTs)", fontsize=11, fontweight='bold', labelpad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.55, zorder=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Legenda fora da área útil no canto superior direito
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', markeredgecolor='#111111', markersize=7, label='Novo Recorde'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()

    if salvar and pasta_saida:
        os.makedirs(pasta_saida, exist_ok=True)
        arq_png = os.path.join(pasta_saida, f"grafico_convergencia_th_{th_pct:02d}.png")
        arq_pdf = os.path.join(pasta_saida, f"grafico_convergencia_th_{th_pct:02d}.pdf")
        plt.savefig(arq_png, dpi=300, bbox_inches='tight')
        plt.savefig(arq_pdf, bbox_inches='tight')
        print(f"  ✔ Salvo Gráfico de Convergência: '{arq_png}' e '{arq_pdf}'")

    return fig


def gerar_grafico_boxplot(distribuicoes_ssim, df_comp=None, th_boxplot="todos", salvar_arquivo="grafico_boxplot_metricas_ssim.png"):
    """
    Função de compatibilidade: gera o boxplot agrupado (todos os thresholds) ou único.
    """
    if th_boxplot not in ("todos", "all", None):
        return gerar_grafico_boxplot_isolado(distribuicoes_ssim, th=float(th_boxplot), tipo_dataset="Teste", salvar_arquivo=salvar_arquivo)

    # Boxplot agrupado
    th_disponiveis = sorted(list({th for (_, th) in distribuicoes_ssim.keys()}))
    metricas_presentes = [m for m in nomes_ssim if any((m, th) in distribuicoes_ssim for th in th_disponiveis)]
    n_groups = len(th_disponiveis)
    n_bars = len(metricas_presentes)
    group_width = 0.82
    bar_width = group_width / n_bars

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    positions_by_metric = {m: [] for m in metricas_presentes}
    data_by_metric = {m: [] for m in metricas_presentes}

    for i, th in enumerate(th_disponiveis):
        center = i + 1
        start = center - (group_width / 2) + (bar_width / 2)
        for j, m in enumerate(metricas_presentes):
            pos = start + j * bar_width
            positions_by_metric[m].append(pos)
            data_by_metric[m].append(distribuicoes_ssim.get((m, th), []))

    for m in metricas_presentes:
        cor = CORES_METRICAS.get(m, '#4a90e2')
        bp = ax.boxplot(
            data_by_metric[m],
            positions=positions_by_metric[m],
            widths=bar_width * 0.84,
            patch_artist=True,
            showmeans=True,
            meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': '#222222', 'markersize': 4.5},
            medianprops={'color': '#111111', 'linewidth': 1.8},
            whiskerprops={'color': '#444444', 'linewidth': 1.1, 'linestyle': '--'},
            capprops={'color': '#444444', 'linewidth': 1.2},
            flierprops={'marker': 'o', 'markerfacecolor': cor, 'markeredgecolor': 'none', 'markersize': 3.5, 'alpha': 0.5}
        )
        for patch in bp['boxes']:
            patch.set_facecolor(cor)
            patch.set_alpha(0.75)
            patch.set_edgecolor('#222222')
            patch.set_linewidth(1.0)

    ax.set_xticks(range(1, n_groups + 1))
    ax.set_xticklabels([f"{int(round(th * 100))}%" for th in th_disponiveis], fontsize=11, fontweight='bold')
    ax.set_xlabel("Threshold de Tolerância de Erro (1 - SSIM)", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Índice de Similaridade Estrutural (SSIM no Teste)", fontsize=11, fontweight='bold', labelpad=10)
    n_amostras = len(next(iter(distribuicoes_ssim.values()))) if distribuicoes_ssim else 30
    ax.set_title(
        f"Robustez e Dispersão de SSIM em Todos os Thresholds (5% a 30%)\n"
        f"[Avaliação Estatística sobre {n_amostras} Imagens]",
        fontsize=12,
        fontweight='bold',
        pad=14,
        ha='center'
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    for x in [i + 0.5 for i in range(1, n_groups)]:
        ax.axvline(x, color='#e0e0e0', linestyle=':', lw=1.2, zorder=0)

    patches_legenda = [
        Patch(facecolor=CORES_METRICAS.get(m, '#999'), edgecolor='#222', alpha=0.75, label=NOMES_LEGIVEIS.get(m, m).split('(')[0].strip())
        for m in metricas_presentes
    ]
    patches_legenda.extend([
        Line2D([0], [0], color='#111111', lw=1.8, label='Mediana'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='#222', markersize=5, label='Média')
    ])
    ax.legend(handles=patches_legenda, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    if salvar_arquivo:
        plt.savefig(salvar_arquivo, dpi=300, bbox_inches='tight')
    return fig


def imprimir_guia_escolha_tcc(df_comp):
    """Exibe no console/notebook a recomendação fundamentada para escolha no TCC."""
    if df_comp.empty:
        return

    print("\n" + "=" * 75)
    print(" 🎓 GUIA DE DECISÃO CIENTÍFICA PARA O TCC")
    print("=" * 75)

    media_econ = df_comp.groupby('Metrica')['Economia_LUTs (%)'].mean()
    menor_desvio = df_comp.groupby('Metrica')['Desvio_Teste'].mean()

    melhor_econ = media_econ.idxmax()
    mais_estavel = menor_desvio.idxmin()

    print(f" 1. Maior Economia Média de LUTs: '{melhor_econ}' ({media_econ[melhor_econ]:.1f}% de economia média)")
    print(f" 2. Maior Uniformidade e Menor Dispersão: '{mais_estavel}' (Menor desvio padrão médio de {menor_desvio[mais_estavel]:.4f})")
    print("-" * 75)
    print(" Recomendações para a escrita da dissertação/monografia:")
    print("  • Se o foco for Economia de Área: defenda 'ssim_medio' ou 'ssim_mediana'")
    print("    pois alcançam máxima redução de LUTs mantendo alta fidelidade perceptual.")
    print("  • Se o foco for Robustez e Equilíbrio Estatístico: defenda 'ssim_mediana' ou 'ssim_mediageometrica'")
    print("    pois atenuam o impacto de imagens atípicas e garantem estabilidade nas bordas.")
    print("=" * 75 + "\n")
