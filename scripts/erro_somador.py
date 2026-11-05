import os
import numpy as np
import pandas as pd

PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_DIR = os.path.join(PROJ_DIR, "results")

def calcula_erro():
    print("Gerando todas as 16.7 milhões de combinações possíveis (12 bits)...")

    csv_luts_path = (
        os.path.join(RESULTS_DIR, "resultados_luts.csv")
        if os.path.exists(os.path.join(RESULTS_DIR, "resultados_luts.csv"))
        else os.path.join(PROJ_DIR, "resultados_luts.csv")
    )
    try:
        df_luts = pd.read_csv(csv_luts_path)
        df_luts['Somador_norm'] = df_luts['Somador'].astype(str).str.replace(' ', '').str.lower()
        dict_luts = dict(zip(df_luts['Somador_norm'], df_luts['LUTs']))
    except FileNotFoundError:
        print("Aviso: 'resultados_luts.csv' não encontrado. Usando LUTs = 0 como fallback.")
        dict_luts = {}

    valores = np.arange(4096, dtype=np.uint16)
    A, B = np.meshgrid(valores, valores)

    soma_exata = A.astype(np.int32) + B.astype(np.int32)
    max_exata = np.max(soma_exata)

    resultados_exato = []
    resultados_trunc0 = []
    resultados_trunc1 = []

    resultados_exato.append({
        'Somador': 'Exato',
        'Bits Truncados': 0,
        'LUTs': dict_luts.get('exato', 19),
        'ED (%)': 0.0,
        'ER (%)': 0.0,
        'Max ED': 0,
        'MED': 0.0,
        'MRED (%)': 0.0
    })

    print("Calculando métricas para os somadores Trunc-0 e Trunc-1 (1 a 5 bits)...")

    # 3. Laço apenas para os 5 bits de truncamento
    for t in range(1, 6):
        mascara_1s = (1 << t) - 1

        # ==========================================
        # CÁLCULO TRUNC 0 (Preenche com Zeros)
        # ==========================================
        soma_aprox_0 = ((A >> t) + (B >> t)) << t
        soma_aprox_0 = soma_aprox_0.astype(np.int32)

        ed_0 = np.abs(soma_exata - soma_aprox_0)
        er_0 = np.mean(ed_0 != 0) * 100
        med_0 = np.mean(ed_0)

        with np.errstate(divide='ignore', invalid='ignore'):
            mred_0 = np.where(soma_exata != 0, ed_0 / soma_exata, 0)
            mred_medio_0 = np.mean(mred_0) * 100

        chave_0 = f'trunc0_{t}'

        resultados_trunc0.append({
            'Somador': f'Trunc0_{t}',
            'Bits Truncados': t,
            'LUTs': dict_luts.get(chave_0, 0),
            'ED (%)': round(np.mean(ed_0) / max_exata * 100, 4),
            'ER (%)': round(er_0, 2),
            'Max ED': np.max(ed_0),
            'MED': round(med_0, 2),
            'MRED (%)': round(mred_medio_0, 4)
        })

        # ==========================================
        # CÁLCULO TRUNC 1 (Preenche com Uns)
        # ==========================================
        soma_aprox_1 = (((A >> t) + (B >> t)) << t) | mascara_1s
        soma_aprox_1 = soma_aprox_1.astype(np.int32)

        ed_1 = np.abs(soma_exata - soma_aprox_1)
        er_1 = np.mean(ed_1 != 0) * 100
        med_1 = np.mean(ed_1)

        with np.errstate(divide='ignore', invalid='ignore'):
            mred_1 = np.where(soma_exata != 0, ed_1 / soma_exata, 0)
            mred_medio_1 = np.mean(mred_1) * 100

        # Nomenclatura ajustada para bater de 1 a 5
        chave_1 = f'trunc1_{t}'

        resultados_trunc1.append({
            'Somador': f'Trunc1_{t}',
            'Bits Truncados': t,
            'LUTs': dict_luts.get(chave_1, 0),
            'ED (%)': round(np.mean(ed_1) / max_exata * 100, 4),
            'ER (%)': round(er_1, 2),
            'Max ED': np.max(ed_1),
            'MED': round(med_1, 2),
            'MRED (%)': round(mred_medio_1, 4)
        })

    # 4. Junta tudo na ordem desejada: Exato -> Trunc0(1 ao 5) -> Trunc1(1 ao 5)
    resultados_finais = resultados_exato + resultados_trunc0 + resultados_trunc1

    # 5. Formata e exporta
    df = pd.DataFrame(resultados_finais)

    print("\n" + "="*80)
    print("   TABELA FINAL DE TRADE-OFF: HARDWARE VS. PRECISÃO (12 BITS)")
    print("="*80)
    print(df.to_string(index=False))

    out_csv = os.path.join(RESULTS_DIR, 'metricas_erro_somadores.csv') if os.path.isdir(RESULTS_DIR) else 'metricas_erro_somadores.csv'
    df.to_csv(out_csv, index=False, decimal=',')
    print(f"\n Resultados salvos em '{out_csv}'")

if __name__ == "__main__":
    calcula_erro()
