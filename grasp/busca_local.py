import random
from grasp.avaliador import calcular_area, obter_funcao_erro, DATASET_TREINO_PADRAO

def busca_local(setup_inicial, config_exata, limite_erro, total_somadores_app, total_somadores_bib, k_vizinhos=None, max_iteracoes_locais=20, dataset_file=DATASET_TREINO_PADRAO, imagem_ref=None, funcao_erro=0, **kwargs):
    if imagem_ref is not None:
        dataset_file = imagem_ref

    if 'tipo_erro' in kwargs:
        funcao_erro = kwargs['tipo_erro']
    elif 'funcao_ssim' in kwargs:
        funcao_erro = kwargs['funcao_ssim']
    elif 'tipo_ssim' in kwargs:
        funcao_erro = kwargs['tipo_ssim']

    if k_vizinhos is not None and k_vizinhos <= 0:
        raise ValueError(f"k_vizinhos deve ser maior que 0 (recebido: {k_vizinhos})")

    fn_erro = obter_funcao_erro(funcao_erro)

    setup_atual = list(setup_inicial)
    solucao_local = list(setup_atual)
    melhoria = True
    iteracao_local = 0
    avaliacoes_ssim = 0
    
    area_atual = calcular_area(setup_atual)
    
    while melhoria and iteracao_local < max_iteracoes_locais:
        melhoria = False
        iteracao_local += 1
        
        for j in range(total_somadores_app):
            peca_original = solucao_local[j]
            
            # Opções disponíveis para troca (excluindo a peça atual)
            opcoes_possiveis = [opcao for opcao in range(total_somadores_bib) if opcao != peca_original]
            
            # Se K foi definido, restringe a vizinhança amostrando K candidatos aleatórios
            if k_vizinhos is not None and k_vizinhos < len(opcoes_possiveis):
                candidatos = random.sample(opcoes_possiveis, k_vizinhos)
            else:
                candidatos = opcoes_possiveis
            
            # Explora a vizinhança restrita
            for opcao in candidatos:
                solucao_local[j] = opcao
                area_viz = calcular_area(solucao_local)
                
                # Otimização: só simula a imagem (SSIM) se a área for promissora (menor que a atual)
                if area_viz < area_atual:
                    avaliacoes_ssim += 1  # Conta apenas avaliação real de SSIM
                    erro_viz = fn_erro(config_exata, solucao_local, dataset_file=dataset_file)
                    
                    # Se o erro estiver dentro do limite, aceitamos a troca
                    if erro_viz <= limite_erro:
                        setup_atual = list(solucao_local)
                        area_atual = area_viz  # Atualiza o novo recorde de área local
                        melhoria = True
                    
            # Restaura a solucao_local para a melhor configuração encontrada até agora
            # antes de testar a vizinhança do próximo somador (j + 1)
            solucao_local[j] = setup_atual[j]
            
    return setup_atual, avaliacoes_ssim