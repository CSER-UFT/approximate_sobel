import os
import time
import pandas as pd
from grasp.construcao import fase_construcao
from grasp.busca_local import busca_local
from grasp.avaliador import calcular_area, obter_funcao_erro, obter_nome_ssim, DATASET_TREINO_PADRAO

TOTAL_AVALIACOES_SSIM = 0
TOTAL_COMBINACOES_GRASP = 0

def obter_total_avaliacoes_ssim():
    """Retorna o número total acumulado de avaliações de SSIM pelo GRASP."""
    global TOTAL_AVALIACOES_SSIM
    return TOTAL_AVALIACOES_SSIM

def obter_total_combinacoes_grasp():
    """Alias de compatibilidade para obter_total_avaliacoes_ssim."""
    return obter_total_avaliacoes_ssim()

def zerar_contador_avaliacoes_ssim():
    """Zera o contador acumulativo de avaliações de SSIM do GRASP."""
    global TOTAL_AVALIACOES_SSIM, TOTAL_COMBINACOES_GRASP
    TOTAL_AVALIACOES_SSIM = 0
    TOTAL_COMBINACOES_GRASP = 0

def zerar_contador_combinacoes_grasp():
    """Alias de compatibilidade para zerar_contador_avaliacoes_ssim."""
    zerar_contador_avaliacoes_ssim()

def grasp_sobel(configuracao_exata, limite_erro, max_iteracoes, alpha, total_somadores_bib, total_somadores_app, k_vizinhos=None, retornar_combinacoes=False, retornar_trace=False, arquivo_trace=None, dataset_file=DATASET_TREINO_PADRAO, imagem_ref=None, funcao_erro=0, **kwargs):
    """
    Executa a meta-heurística GRASP para seleção de somadores aproximados.

    Retorno:
        Se retornar_trace=False e retornar_combinacoes=False:
            tuple: (setup_final, area_final, respeitou_limite)
        Se retornar_trace=False e retornar_combinacoes=True:
            tuple: (setup_final, area_final, avaliacoes_desta_busca, iter_final, respeitou_limite)
        Se retornar_trace=True:
            adiciona historico_trace (list de dicts com o registro por iteração) ao final da tupla.

        Onde:
            respeitou_limite (bool): True se a solução encontrada respeitou o limite_erro;
                                     False caso nenhuma solução viável tenha sido encontrada,
                                     retornando o melhor setup observado (menor erro).
    """
    global TOTAL_AVALIACOES_SSIM, TOTAL_COMBINACOES_GRASP
    if imagem_ref is not None:
        dataset_file = imagem_ref

    if 'tipo_erro' in kwargs:
        funcao_erro = kwargs['tipo_erro']
    elif 'funcao_ssim' in kwargs:
        funcao_erro = kwargs['funcao_ssim']
    elif 'tipo_ssim' in kwargs:
        funcao_erro = kwargs['tipo_ssim']

    fn_erro = obter_funcao_erro(funcao_erro)
    nome_erro = obter_nome_ssim(fn_erro)

    melhor_setup_global = None
    melhor_area_global = float('inf')
    melhor_erro_global = float('inf')
    melhor_iteracao_global = None

    # Fallback para caso nenhuma iteração respeite o limite de erro
    melhor_setup_inviavel = None
    melhor_area_inviavel = float('inf')
    melhor_erro_inviavel = float('inf')
    melhor_iteracao_inviavel = None

    avaliacoes_desta_busca = 0
    historico_trace = []

    for iteracao in range(max_iteracoes):
        t0_iter = time.time()
        
        # 1. Fase de Construção (retorna avaliações reais de SSIM)
        setup_construido, n_const = fase_construcao(
            configuracao_exata, limite_erro, alpha, total_somadores_bib, total_somadores_app,
            dataset_file=dataset_file, funcao_erro=fn_erro
        )
        area_construida = calcular_area(setup_construido)
        avaliacoes_desta_busca += n_const
        TOTAL_AVALIACOES_SSIM += n_const
        TOTAL_COMBINACOES_GRASP += n_const
        
        # 2. Busca Local (Refinamento - retorna avaliações reais de SSIM)
        setup_refinado, n_local = busca_local(
            setup_construido, configuracao_exata, limite_erro, total_somadores_app, total_somadores_bib,
            k_vizinhos=k_vizinhos, dataset_file=dataset_file, funcao_erro=fn_erro
        )
        avaliacoes_desta_busca += n_local
        TOTAL_AVALIACOES_SSIM += n_local
        TOTAL_COMBINACOES_GRASP += n_local
        
        # 3. Avaliação Final
        area_final = calcular_area(setup_refinado)
        avaliacoes_desta_busca += 1
        TOTAL_AVALIACOES_SSIM += 1
        TOTAL_COMBINACOES_GRASP += 1
        erro_final = fn_erro(configuracao_exata, setup_refinado, dataset_file=dataset_file)
        
        # Atualiza melhor fallback inviável (menor erro)
        bateu_recorde_inviavel = (erro_final < melhor_erro_inviavel) or \
                                 (erro_final == melhor_erro_inviavel and area_final < melhor_area_inviavel)
        if bateu_recorde_inviavel:
            melhor_setup_inviavel = list(setup_refinado)
            melhor_area_inviavel = area_final
            melhor_erro_inviavel = erro_final
            melhor_iteracao_inviavel = iteracao + 1

        # 4. Atualiza se bateu o recorde de área OU empatou em área com melhor qualidade visual
        bateu_recorde = False
        if erro_final <= limite_erro:
            bateu_recorde = (area_final < melhor_area_global) or \
                            (area_final == melhor_area_global and erro_final < melhor_erro_global)

            if bateu_recorde:
                motivo = "Menor Área" if area_final < melhor_area_global else "Desempate por Qualidade Visual (Menor Erro)"
                melhor_area_global = area_final
                melhor_erro_global = erro_final
                melhor_setup_global = list(setup_refinado)
                melhor_iteracao_global = iteracao + 1
                print(f"Iteração {iteracao+1}/{max_iteracoes}: Novo recorde! [{motivo}] Área = {melhor_area_global} LUTs | Erro ({nome_erro}) = {erro_final:.4f} | Avaliações SSIM nesta busca: {avaliacoes_desta_busca} (Total acumulado: {TOTAL_AVALIACOES_SSIM})")

        duracao_iter = round(time.time() - t0_iter, 2)
        viavel = (erro_final <= limite_erro)

        linha_trace = {
            'Iteracao': iteracao + 1,
            'Setup_Construido': str(setup_construido),
            'Area_Construida': area_construida,
            'Setup_Refinado': str(setup_refinado),
            'Area_Refinada': area_final,
            'Melhoria_Area_Local': area_construida - area_final,
            'Erro_Final': round(float(erro_final), 5),
            'Limite_Erro': limite_erro,
            'Viavel': viavel,
            'Bateu_Recorde': bateu_recorde,
            'Melhor_Area_Global': melhor_area_global if melhor_area_global != float('inf') else None,
            'Melhor_Erro_Global': round(float(melhor_erro_global), 5) if melhor_erro_global != float('inf') else None,
            'Melhor_Setup_Global': str(melhor_setup_global) if melhor_setup_global is not None else None,
            'Avaliacoes_Construcao': n_const,
            'Avaliacoes_Busca_Local': n_local,
            'Avaliacoes_Iteracao': n_const + n_local + 1,
            'Avaliacoes_Acumuladas': avaliacoes_desta_busca,
            'Tempo_Iteracao_s': duracao_iter
        }
        historico_trace.append(linha_trace)

    if melhor_setup_global is not None:
        respeitou_limite = True
        setup_retorno = melhor_setup_global
        area_retorno = melhor_area_global
        iter_retorno = melhor_iteracao_global
    else:
        respeitou_limite = False
        setup_retorno = melhor_setup_inviavel if melhor_setup_inviavel is not None else list(configuracao_exata)
        area_retorno = melhor_area_inviavel if melhor_area_inviavel != float('inf') else calcular_area(setup_retorno)
        iter_retorno = melhor_iteracao_inviavel if melhor_iteracao_inviavel is not None else 0

    if arquivo_trace:
        os.makedirs(os.path.dirname(os.path.abspath(arquivo_trace)), exist_ok=True)
        pd.DataFrame(historico_trace).to_csv(arquivo_trace, index=False)

    if retornar_combinacoes and retornar_trace:
        return setup_retorno, area_retorno, avaliacoes_desta_busca, iter_retorno, respeitou_limite, historico_trace
    elif retornar_combinacoes:
        return setup_retorno, area_retorno, avaliacoes_desta_busca, iter_retorno, respeitou_limite
    elif retornar_trace:
        return setup_retorno, area_retorno, respeitou_limite, historico_trace
    return setup_retorno, area_retorno, respeitou_limite