import random
from grasp.avaliador import calcular_area, obter_funcao_erro, DATASET_TREINO_PADRAO

def fase_construcao(config_exata, limite_erro, alpha, total_somadores_bib, total_somadores_app, dataset_file=DATASET_TREINO_PADRAO, imagem_ref=None, funcao_erro=0, **kwargs):
    if imagem_ref is not None:
        dataset_file = imagem_ref

    if 'tipo_erro' in kwargs:
        funcao_erro = kwargs['tipo_erro']
    elif 'funcao_ssim' in kwargs:
        funcao_erro = kwargs['funcao_ssim']
    elif 'tipo_ssim' in kwargs:
        funcao_erro = kwargs['tipo_ssim']

    fn_erro = obter_funcao_erro(funcao_erro)

    # Inicializa todos os somadores com a opção 0 (Somador Exato)
    setup_atual = [0] * total_somadores_app
    avaliacoes_ssim = 0
    
    for j in range(total_somadores_app):
        lista_candidatos = []
        
        # Testa cada somador da biblioteca (0 a 10) para a posição j
        for opcao in range(total_somadores_bib):
            setup_atual[j] = opcao
            area_atual = calcular_area(setup_atual)
            
            avaliacoes_ssim += 1  # Conta avaliação real de SSIM
            erro_atual = fn_erro(config_exata, setup_atual, dataset_file=dataset_file)
            
            if erro_atual <= limite_erro:
                # Salva a tupla (ID do somador, Custo de Área)
                lista_candidatos.append((opcao, area_atual))
        
        # Se nenhuma opção aproximada servir, volta para o Exato de forma segura
        if not lista_candidatos:
            setup_atual[j] = 0
            continue
            
        # Ordena os candidatos do menor custo de área para o maior
        lista_candidatos.sort(key=lambda x: x[1])
        
        # O custo é o segundo elemento da tupla (índice 1)
        c_min = lista_candidatos[0][1]
        c_max = lista_candidatos[-1][1]
        
        # Calcula a nota de corte usando o fator alpha
        limite = c_min + alpha * (c_max - c_min)
        
        # Cria a Lista Restrita de Candidatos (RCL) filtrando quem está dentro do limite
        rcl = [cand for cand in lista_candidatos if cand[1] <= limite]
        
        if not rcl:
            setup_atual[j] = 0
        else:
            # Sorteia um candidato válido da RCL e aplica na solução
            escolha = random.choice(rcl)
            setup_atual[j] = escolha[0] # escolha[0] é o 'opcao' (ID do somador)
            
    return setup_atual, avaliacoes_ssim