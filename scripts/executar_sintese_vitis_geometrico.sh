#!/usr/bin/env bash
# ==============================================================================
# executar_sintese_vitis_geometrico.sh
# Script para chamar o Vitis HLS e sintetizar o Full Design (GEOMETRICO)
# Baseline (Tudo Exato) e Aproximações GRASP (5% a 30%)
# ==============================================================================

set -e

echo '=================================================================='
echo '    SÍNTESE VITIS HLS: SOBEL FULL DESIGN [GEOMETRICO]    '
echo '=================================================================='
echo "Data/Hora: $(date)"
echo "Diretório: $(pwd)"
echo '------------------------------------------------------------------'

# 1. Configuração do ambiente Xilinx Vitis
if command -v vitis_hls >/dev/null 2>&1; then
    echo "✔ vitis_hls encontrado no PATH ($(which vitis_hls))."
else
    SETTINGS_FOUND=""
    POSSIBLE_PATHS=(
        '/tools/Xilinx/Vitis_HLS/2022.2/settings64.sh'
        '/tools/Xilinx/Vitis/2022.2/settings64.sh'
        '/tools/Xilinx/Vivado/2022.2/settings64.sh'
        '/tools/Xilinx/Vitis_HLS/2022.1/settings64.sh'
        '/tools/Xilinx/Vitis/2022.1/settings64.sh'
        '/tools/Xilinx/Vivado/2022.1/settings64.sh'
        '/opt/Xilinx/Vitis_HLS/2022.2/settings64.sh'
        '/opt/Xilinx/Vitis/2022.2/settings64.sh'
        '/opt/Xilinx/Vivado/2022.2/settings64.sh'
    )
    for p in "${POSSIBLE_PATHS[@]}"; do
        if [ -f "$p" ]; then
            SETTINGS_FOUND="$p"
            break
        fi
    done

    if [ -n "$SETTINGS_FOUND" ]; then
        echo "✔ Carregando variáveis do ambiente Xilinx: $SETTINGS_FOUND"
        source "$SETTINGS_FOUND"
    else
        echo "❌ ERRO: 'vitis_hls' não encontrado no PATH nem nos diretórios padrão."
        echo "   Execute 'source /caminho/para/settings64.sh' ou rode no servidor com Vitis instalado."
        exit 1
    fi
fi

# 2. Executa a síntese em lote via TCL
TCL_SCRIPT='scripts/sintese_sobel_geometrico.tcl'
if [ ! -f "$TCL_SCRIPT" ] && [ -f 'sintese_sobel_geometrico.tcl' ]; then
    TCL_SCRIPT='sintese_sobel_geometrico.tcl'
fi
if [ ! -f "$TCL_SCRIPT" ]; then
    echo "❌ ERRO: Script TCL '$TCL_SCRIPT' não encontrado!"
    exit 1
fi

echo ''
echo '>>> Iniciando síntese em lote no Vitis HLS...'
vitis_hls -f "$TCL_SCRIPT"

echo ''
echo '=================================================================='
echo "✔ Síntese finalizada com sucesso! Relatórios salvos em 'reports_sintese_geometrico/'"
echo '=================================================================='
