import os
import xml.etree.ElementTree as ET
import pandas as pd

PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_DIR = os.path.join(PROJ_DIR, "results")

def coletar():
    print("Iniciando extração de LUTs dos relatórios do Vitis HLS...")

    somadores = ['add_exact']

    # Busca do trunc0_1 ao trunc0_5
    for i in range(1, 6):
        somadores.append(f'add_trunc0_{i}')
    # Busca do trunc1_1 ao trunc1_5 (AQUI ESTAVA O ERRO)
    for i in range(1, 6):
        somadores.append(f'add_trunc1_{i}')

    resultados = []

    for nome in somadores:
        caminho_xml_proj = os.path.join(PROJ_DIR, f"proj_somadores/sol_{nome}/syn/report/{nome}_csynth.xml")
        caminho_xml_res = os.path.join(RESULTS_DIR, f"reports_sintese_somadores/{nome}_csynth.xml")
        caminho_xml = caminho_xml_proj if os.path.exists(caminho_xml_proj) else caminho_xml_res

        luts = 0
        if os.path.exists(caminho_xml):
            tree = ET.parse(caminho_xml)
            root = tree.getroot()

            luts_node = root.find('.//AreaEstimates/Resources/LUT')
            if luts_node is not None:
                luts = int(luts_node.text)
            print(f"[{nome}] Encontrado: {luts} LUTs")
        else:
            print(f"[{nome}] AVISO: XML não encontrado. Caminho tentado: {caminho_xml}")

        nome_csv = nome.replace("add_", "")
        resultados.append({'Somador': nome_csv, 'LUTs': luts})

    df = pd.DataFrame(resultados)
    out_csv = os.path.join(RESULTS_DIR, 'resultados_luts.csv') if os.path.isdir(RESULTS_DIR) else 'resultados_luts.csv'
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Arquivo '{out_csv}' gerado com sucesso!")

if __name__ == "__main__":
    coletar()
