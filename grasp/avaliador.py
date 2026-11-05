"""
avaliador.py

Módulo avaliador para o GRASP e DSE.
Executa a simulação do Sobel em hardware via subprocesso C++ (OpenMP + Vitis HLS),
calcula o erro visual (SSIM) e o consumo de área (LUTs).
"""

import os
import csv
import subprocess
import tempfile
import atexit
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Caminhos base
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SOBEL_BIN = os.path.join(PROJ_DIR, "sobel")
SOBEL_CPP = os.path.join(PROJ_DIR, "sobel.cpp")
_CSV_LUTS = (
    os.path.join(PROJ_DIR, "results", "resultados_luts.csv")
    if os.path.isfile(os.path.join(PROJ_DIR, "results", "resultados_luts.csv"))
    else os.path.join(PROJ_DIR, "resultados_luts.csv")
)

# Datasets padrão
DATASET_TREINO_PADRAO = "training.in"
DATASET_TESTE_PADRAO = "testing.in"

# Nomes padronizados dos 11 somadores da biblioteca
nomes_somadores = [
    'exact',
    'trunc0_1', 'trunc0_2', 'trunc0_3', 'trunc0_4', 'trunc0_5',
    'trunc1_1', 'trunc1_2', 'trunc1_3', 'trunc1_4', 'trunc1_5'
]

# Caches globais
_cache_luts = None
_cache_refs = {}          # (dataset_original, dataset_resolvido_abs) -> (list[caminhos_ref], list[np.ndarray])
_cache_valores_ssim = {}  # (tuple(setup), dataset_path) -> list[float]
_cache_erros = {}         # (tuple(setup), dataset_path, metrica) -> float

# Gerenciamento de arquivos temporários para imagens únicas (Thread-safe)
_arquivos_temp = []
_cache_temp_datasets = {}


def _limpar_arquivos_temp():
    """Remove os arquivos temporários criados durante a execução."""
    for f in _arquivos_temp:
        if os.path.isfile(f):
            try:
                os.remove(f)
            except OSError:
                pass


atexit.register(_limpar_arquivos_temp)


# ==============================================================================
# 1. Simulação C++ via Subprocess
# ==============================================================================

def garantir_binario_sobel():
    """Garante que o executável C++ compilado com OpenMP existe."""
    if not os.path.isfile(SOBEL_BIN):
        include_dir = os.path.join(PROJ_DIR, "include")
        cmd = ["g++", "-O3", "-fopenmp", "-I", include_dir, SOBEL_CPP, "-o", SOBEL_BIN]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Erro ao compilar {SOBEL_CPP}:\n{res.stderr}")
    return SOBEL_BIN


def _resolver_dataset(dataset_file):
    """Resolve o caminho absoluto do arquivo .in ou cria um para imagem única de forma thread-safe."""
    caminho = dataset_file if os.path.isabs(dataset_file) else os.path.join(PROJ_DIR, dataset_file)
    if not os.path.isfile(caminho):
        caminho_alt = os.path.join(PROJ_DIR, os.path.basename(dataset_file))
        if os.path.isfile(caminho_alt):
            caminho = caminho_alt
        else:
            raise FileNotFoundError(f"Dataset não encontrado: {dataset_file}")

    if caminho.endswith((".ppm", ".png", ".jpg")):
        caminho_abs = os.path.abspath(caminho)
        if caminho_abs in _cache_temp_datasets and os.path.isfile(_cache_temp_datasets[caminho_abs]):
            return _cache_temp_datasets[caminho_abs]

        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False, dir=PROJ_DIR)
        tf.write(f"1\n{caminho_abs}\n")
        tf.flush()
        tf.close()
        tmp_in = tf.name
        _arquivos_temp.append(tmp_in)
        _cache_temp_datasets[caminho_abs] = tmp_in
        return tmp_in
    return caminho


def executar_sobel(setup, dataset_file=DATASET_TREINO_PADRAO, kind="cand"):
    """
    Executa o binário C++ via subprocess com a combinação de 5 somadores.
    Retorna a lista de caminhos absolutos das imagens geradas.
    """
    garantir_binario_sobel()
    arquivo_in = _resolver_dataset(dataset_file)

    if isinstance(setup, dict):
        a = [int(setup.get(f'a{i}', 0)) for i in range(5)]
    else:
        a = [int(x) for x in setup[:5]]

    cmd = [SOBEL_BIN, arquivo_in, kind] + [str(x) for x in a]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Erro na execução do Sobel C++ (código {res.returncode}):\n{res.stderr}")

    arquivos = res.stdout.strip().split()
    return [f if os.path.isabs(f) else os.path.join(PROJ_DIR, f) for f in arquivos]


def obter_referencias_exatas(dataset_file=DATASET_TREINO_PADRAO):
    """Gera o baseline exato [0, 0, 0, 0, 0] no C++ e o mantém em RAM."""
    caminho_in = _resolver_dataset(dataset_file)
    chave = (dataset_file, os.path.abspath(caminho_in))
    if chave in _cache_refs:
        return _cache_refs[chave]

    arquivos_ref = executar_sobel([0, 0, 0, 0, 0], caminho_in, kind="precise")
    imagens_ref = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in arquivos_ref]

    _cache_refs[chave] = (arquivos_ref, imagens_ref)
    return _cache_refs[chave]


# ==============================================================================
# 2. Tabela de LUTs e Área
# ==============================================================================

def carregar_tabela_luts(caminho_csv=_CSV_LUTS):
    """Carrega o mapeamento de ID do somador para quantidade de LUTs."""
    global _cache_luts
    if _cache_luts is not None:
        return _cache_luts

    tabela = {}
    if os.path.isfile(caminho_csv):
        with open(caminho_csv, newline='', encoding='utf-8') as f:
            leitor = csv.reader(f)
            next(leitor, None)  # Pula cabeçalho
            idx = 0
            for row in leitor:
                if len(row) >= 2 and row[1].strip().isdigit():
                    tabela[idx] = int(row[1].strip())
                    idx += 1

    assert len(tabela) == len(nomes_somadores), (
        f"Esperado {len(nomes_somadores)} somadores na tabela de LUTs, mas foram carregados {len(tabela)}"
    )

    _cache_luts = tabela
    return tabela


tabela_luts = carregar_tabela_luts()


def calcular_area(setup):
    """Calcula a área total em LUTs de uma combinação de somadores."""
    tab = carregar_tabela_luts()
    if not tab:
        raise RuntimeError("Tabela de LUTs vazia — verifique resultados_luts.csv")

    if isinstance(setup, dict):
        total = 0
        for k, v in setup.items():
            if k.startswith('a'):
                if v not in tab:
                    raise KeyError(f"Somador ID {v} não encontrado na tabela de LUTs")
                total += tab[v]
        return total

    total = 0
    for s in setup:
        s_int = int(s)
        if s_int not in tab:
            raise KeyError(f"Somador ID {s_int} não encontrado na tabela de LUTs")
        total += tab[s_int]
    return total


# ==============================================================================
# 3. Métricas e Avaliação de Erro (SSIM)
# ==============================================================================

def obter_valores_ssim(setup, dataset_file=DATASET_TREINO_PADRAO, imagem_ref=None):
    """Calcula os valores de SSIM de todas as imagens do dataset em paralelo."""
    alvo = imagem_ref or dataset_file
    chave = (tuple(setup[:5]), alvo)
    if chave in _cache_valores_ssim:
        return _cache_valores_ssim[chave]

    if all(s == 0 for s in setup[:5]):
        _, imgs_ref = obter_referencias_exatas(alvo)
        res = [1.0] * len(imgs_ref)
        _cache_valores_ssim[chave] = res
        return res

    arquivos_cand = executar_sobel(setup, alvo, kind="cand")
    _, imgs_ref = obter_referencias_exatas(alvo)

    def _calc_ssim(par):
        img_ref, path_cand = par
        img_cand = cv2.imread(path_cand, cv2.IMREAD_GRAYSCALE)
        return float(ssim(img_ref, img_cand, data_range=255))

    with ThreadPoolExecutor(max_workers=8) as pool:
        valores_ssim = list(pool.map(_calc_ssim, zip(imgs_ref, arquivos_cand)))

    _cache_valores_ssim[chave] = valores_ssim
    return valores_ssim


def ssim_medio(config_exata=None, setup=None, dataset_file=DATASET_TREINO_PADRAO, imagem_ref=None, **kwargs):
    """Erro baseado na média aritmética: 1.0 - média(SSIM)."""
    s = setup if setup is not None else config_exata
    alvo = imagem_ref or dataset_file
    chave = (tuple(s[:5]), alvo, 'ssim_medio')
    if chave in _cache_erros:
        return _cache_erros[chave]
    if all(x == 0 for x in s[:5]):
        return 0.0

    vals = obter_valores_ssim(s, alvo)
    erro = float(max(0.0, 1.0 - np.mean(vals))) if vals else 1.0
    _cache_erros[chave] = erro
    return erro


def ssim_mediana(config_exata=None, setup=None, dataset_file=DATASET_TREINO_PADRAO, imagem_ref=None, **kwargs):
    """Erro baseado na mediana: 1.0 - mediana(SSIM)."""
    s = setup if setup is not None else config_exata
    alvo = imagem_ref or dataset_file
    chave = (tuple(s[:5]), alvo, 'ssim_mediana')
    if chave in _cache_erros:
        return _cache_erros[chave]
    if all(x == 0 for x in s[:5]):
        return 0.0

    vals = obter_valores_ssim(s, alvo)
    erro = float(max(0.0, 1.0 - np.median(vals))) if vals else 1.0
    _cache_erros[chave] = erro
    return erro


def ssim_mediageometrica(config_exata=None, setup=None, dataset_file=DATASET_TREINO_PADRAO, imagem_ref=None, **kwargs):
    """Erro baseado na média geométrica: 1.0 - geom(SSIM)."""
    s = setup if setup is not None else config_exata
    alvo = imagem_ref or dataset_file
    chave = (tuple(s[:5]), alvo, 'ssim_mediageometrica')
    if chave in _cache_erros:
        return _cache_erros[chave]
    if all(x == 0 for x in s[:5]):
        return 0.0

    vals = obter_valores_ssim(s, alvo)
    if not vals:
        return 1.0
    arr = np.array(vals, dtype=np.float64)
    geom = float(np.exp(np.mean(np.log(np.maximum(arr, 1e-10)))))
    erro = float(max(0.0, 1.0 - geom))
    _cache_erros[chave] = erro
    return erro


def ssim_min(config_exata=None, setup=None, dataset_file=DATASET_TREINO_PADRAO, imagem_ref=None, **kwargs):
    """Erro baseado no pior caso (mínimo SSIM): 1.0 - min(SSIM)."""
    s = setup if setup is not None else config_exata
    alvo = imagem_ref or dataset_file
    chave = (tuple(s[:5]), alvo, 'ssim_min')
    if chave in _cache_erros:
        return _cache_erros[chave]
    if all(x == 0 for x in s[:5]):
        return 0.0

    vals = obter_valores_ssim(s, alvo)
    erro = float(max(0.0, 1.0 - np.min(vals))) if vals else 1.0
    _cache_erros[chave] = erro
    return erro


# Estrutura de métricas
nomes_ssim = ['ssim_medio', 'ssim_mediana', 'ssim_mediageometrica', 'ssim_min']
FUNCOES_SSIM = [ssim_medio, ssim_mediana, ssim_mediageometrica, ssim_min]

_MAPA_METRICAS = {
    '0': ssim_medio, 'ssim_medio': ssim_medio, 'medio': ssim_medio, 'media': ssim_medio,
    '1': ssim_mediana, 'ssim_mediana': ssim_mediana, 'mediana': ssim_mediana,
    '2': ssim_mediageometrica, 'ssim_mediageometrica': ssim_mediageometrica, 'geometrico': ssim_mediageometrica,
    '3': ssim_min, 'ssim_min': ssim_min, 'min': ssim_min, 'minimo': ssim_min
}


def obter_funcao_erro(opcao_ou_fn=0):
    """Retorna a função de erro a partir de um índice numérico (0..3), nome ou callable."""
    if opcao_ou_fn is None:
        return ssim_medio
    if callable(opcao_ou_fn):
        return opcao_ou_fn
    if isinstance(opcao_ou_fn, int):
        if not (0 <= opcao_ou_fn < len(FUNCOES_SSIM)):
            indices_validos = list(range(len(FUNCOES_SSIM)))
            raise ValueError(
                f"Índice de métrica inválido: {opcao_ou_fn}. "
                f"Índices válidos são {indices_validos} ({nomes_ssim})"
            )
        return FUNCOES_SSIM[opcao_ou_fn]
    if isinstance(opcao_ou_fn, str) and opcao_ou_fn.lower() in _MAPA_METRICAS:
        return _MAPA_METRICAS[opcao_ou_fn.lower()]
    raise ValueError(f"Métrica SSIM desconhecida: {opcao_ou_fn}")


def obter_nome_ssim(opcao_ou_fn):
    """Retorna o nome da função SSIM informada."""
    fn = obter_funcao_erro(opcao_ou_fn)
    for nome, f in zip(nomes_ssim, FUNCOES_SSIM):
        if fn is f:
            return nome
    return getattr(fn, '__name__', str(fn))


def calcular_ssim_metrica(opcao_ou_fn, setup, dataset_file=DATASET_TESTE_PADRAO, imagem_ref=None):
    """Retorna o valor da métrica SSIM (0 a 1)."""
    fn = obter_funcao_erro(opcao_ou_fn)
    erro = fn(setup=setup, dataset_file=dataset_file, imagem_ref=imagem_ref)
    return float(max(0.0, 1.0 - erro))


def avaliar_configuracao(setup, dataset_file=DATASET_TESTE_PADRAO, imagem_ref=None, funcao_erro=0, kind="cand"):
    """
    Avalia a configuração no dataset especificado (treino ou teste).
    Retorna a tupla: (erro, ssim_metrica, mse_medio, psnr_medio).
    """
    alvo = imagem_ref or dataset_file
    _, imgs_ref = obter_referencias_exatas(alvo)
    arquivos_cand = executar_sobel(setup, alvo, kind=kind)

    def _worker(par):
        img_ref, f_cand = par
        img_aprox = cv2.imread(f_cand, cv2.IMREAD_GRAYSCALE)
        val_ssim = float(ssim(img_ref, img_aprox, data_range=255))
        val_mse = float(np.mean((img_ref.astype(np.float64) - img_aprox.astype(np.float64)) ** 2))
        val_psnr = float(cv2.PSNR(img_ref, img_aprox)) if val_mse > 0 else float('inf')
        return val_ssim, val_mse, val_psnr

    with ThreadPoolExecutor(max_workers=8) as pool:
        resultados = list(pool.map(_worker, zip(imgs_ref, arquivos_cand)))

    valores_ssim = [r[0] for r in resultados]
    valores_mse = [r[1] for r in resultados]
    valores_psnr = [r[2] for r in resultados]

    fn = obter_funcao_erro(funcao_erro)
    nome = obter_nome_ssim(fn)

    if nome == 'ssim_mediana':
        ssim_val = float(np.median(valores_ssim))
    elif nome == 'ssim_mediageometrica':
        arr = np.array(valores_ssim, dtype=np.float64)
        ssim_val = float(np.exp(np.mean(np.log(np.maximum(arr, 1e-10)))))
    elif nome == 'ssim_min':
        ssim_val = float(np.min(valores_ssim))
    else:
        ssim_val = float(np.mean(valores_ssim))

    mse_medio = float(np.mean(valores_mse))
    valores_psnr_validos = [p for p in valores_psnr if not np.isinf(p)]
    psnr_medio = float(np.mean(valores_psnr_validos)) if valores_psnr_validos else float('inf')
    erro = float(max(0.0, 1.0 - ssim_val))
    return erro, ssim_val, mse_medio, psnr_medio


def avaliar_no_treino(setup, dataset_file=DATASET_TREINO_PADRAO, imagem_ref=None, funcao_erro=0):
    """
    Avalia a configuração no conjunto de treino.
    Retorna a tupla: (erro_treino, ssim_treino, mse_treino, psnr_treino).
    """
    return avaliar_configuracao(
        setup,
        dataset_file=dataset_file,
        imagem_ref=imagem_ref,
        funcao_erro=funcao_erro,
        kind="treino_cand"
    )


def avaliar_no_teste(setup, dataset_file=DATASET_TESTE_PADRAO, imagem_ref=None, funcao_erro=0):
    """
    Avalia a configuração no conjunto de teste.
    Retorna a tupla: (erro_teste, ssim_teste, mse_teste, psnr_teste).
    """
    return avaliar_configuracao(
        setup,
        dataset_file=dataset_file,
        imagem_ref=imagem_ref,
        funcao_erro=funcao_erro,
        kind="teste_cand"
    )


# ==============================================================================
# 4. Geração de Imagens e Compatibilidade
# ==============================================================================

def gerar_ppm_threshold(setup, threshold, dataset_file=DATASET_TESTE_PADRAO, imagem_ref=None):
    """Gera as imagens finais identificadas pelo threshold (ex: 71.approx_th_05.sobel.ppm)."""
    alvo = imagem_ref or dataset_file
    th_tag = int(round(threshold * 100))
    return executar_sobel(setup, alvo, kind=f"approx_th_{th_tag:02d}")


def sobel_kernel(setup, input_image):
    """Emulador para teste de imagem única usando o C++."""
    tmp_in = os.path.join(PROJ_DIR, ".single_kernel_test.ppm")
    cv2.imwrite(tmp_in, input_image)
    out_files = executar_sobel(setup, tmp_in, kind="kernel_out")
    res = cv2.imread(out_files[0], cv2.IMREAD_GRAYSCALE)
    for f in [tmp_in, os.path.join(PROJ_DIR, ".single_input.in"), out_files[0]]:
        if os.path.isfile(f):
            try: os.remove(f)
            except OSError: pass
    return res


# Aliases essenciais
calcular_erro = ssim_medio
ssim_geometrico = ssim_mediageometrica
ssim_minimo = ssim_min
sobel_subprocess = executar_sobel
FUNCOES_SOMADORES = []
