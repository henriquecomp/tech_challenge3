import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURAÇÃO INICIAL E CARREGAMENTO DOS ARTEFATOS ---

# @st.cache_resource garante que o modelo e os outros arquivos pesados
# sejam carregados apenas uma vez, tornando a aplicação mais rápida.
@st.cache_resource
def carregar_artefatos():
    """ Carrega o modelo, o encoder e as colunas de treino salvos. """
    try:
        modelo = joblib.load('modelo_rf_binario.joblib')
        le_alvo = joblib.load('label_encoder_alvo_binario_rf.joblib')
        colunas_treino = joblib.load('colunas_treino_binario_rf.joblib')
    except FileNotFoundError:
        st.error(
            "Erro: Arquivos do modelo não encontrados! "
            "Certifique-se de que 'modelo_rf_binario.joblib', "
            "'label_encoder_alvo_binario_rf.joblib' e 'colunas_treino_binario_rf.joblib' "
            "estão na mesma pasta que o script."
        )
        return None, None, None
    return modelo, le_alvo, colunas_treino

# Carregar os artefatos na inicialização da aplicação
modelo_rf, le_alvo, colunas_treino = carregar_artefatos()

# --- INTERFACE DA APLICAÇÃO ---

# Título principal da página
st.title('Previsão de Acidentes de Trânsito com Random Forest')

# Abas para organizar o conteúdo
tab_descritivo, tab_analise, tab_distribuicao, tab_correlacao, tab_modelo, tab_previsao = st.tabs(["Tech challenge", "Análise dados por ano", "% distribuição ano", "Correlações", "Modelo", "Simulador de Previsão"])

# --- ABA 1: DESCRITIVO DO TECH CHALLENGE ---
with tab_descritivo:
    st.subheader('Qual é o problema?')
    st.info("""
    **Pergunta de Negócio:** Durante um acidente numa rodovia federal na região metropolitana de Recife, qual seria a classificação do acidente: **COM VÍTIMAS ou SEM VÍTIMAS?**

    **Objetivo:** Otimizar a alocação de recursos do estado (ambulâncias, polícias), reduzindo custos e excesso de trabalho em acidentes de menor gravidade.
    """)

    st.subheader('Coleta de dados')
    st.info("""
    Foram utilizados os dados extraídos do portal dados abertos do governo federal no link
    https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
    """)

    st.subheader('Armazenamento')
    st.info("""
    Arquivo estruturado csv com separador ;
    """)

    st.subheader('Quais são os comportamentos dos dados?')
    st.info("""
    * Existe um forte desequilíbrio de classes. A proporção é de 10 (COM VÍTIMAS) para 1 (SEM VÍTIMAS), isto faz com que o modelo tenha dificuldades em aprender os padrões de classes minoritárias.
            
    * Os padrões dos acidentes não são constantes ao longo do tempo. Ao treinar com um ano e validar com o outro mostra que os tipos de acidente mudam de um ano para outro. Isso mostra o comportamento dos dados e a principal razão pela qual a validação temporal foi crucial.
    """)

    st.subheader('Quais são as particularidades?')
    st.info("""
    * Necessidade de simplificação dos problemas, afunilando os dados com mínimo possível para um modelo mais estável e confiável.
            
    * Ao simplificar as classes em apenas 2 (COM VITIMA e SEM VITIMA) passamos ter uma classificação binária fazendo com que o modelo alcançasse performance superiores.
            
    * Alta cardinalidade em features geográficas. O local do acidente possuía muitos valores únicos (alta cardinalidade). Então agrupamos por 50 trechos mais críticos das rodovias e os demais classificamos como OUTROS.
            
    * Necessidade de engenharia de feature para chegar ao objetivo.
    """)

    st.subheader('Como são as distribuições?')
    st.info("""
    * A classe é muito assimétrica e a classe “COM VITIMA” domina o conjunto de dados.
            
    * Distribuição das features não são estáveis ao longo do tempo. A proporção de colisões traseiras, por exemplo, são diferentes em 2023 e 2024. Isso reforça a necessidade de validação temporal.
    """)

    st.subheader('Como são as correlações entre as variáveis?')
    st.info("""
    * As correlações entre as variáveis são muito baixas.
    """)

    st.subheader('Existe sazonalidade?')
    st.info("""
    * A sazonalidade é um fator muito forte nos dados. A sua existência são comprovada por duas features fase_dia e final_semana. Sendo que a distribuição dos acidentes mudam drasticamente do DIA para a NOITE e o comportamento do trânsito e, consequentemente, dos acidentes é diferente durantes os final de semana.
            
    * Meses como janeiro, julho e dezembro por ser férias e feriados prolongados podem influenciar nos tipos de acidente.
    """)

    st.subheader('Análise estatísticas (médias, medianas, desvios, outliers etc)?')
    st.info("""
    * Como reduzimos muito os dados para chegar em um modelo com a máxima eficiência, temos somente uma coluna numérica que é o ano.
            
    * Foram encontrados outliers como em idade onde existiam idades que era o ano.
    """)

    st.subheader('Processamento dos dados')
    st.info("""
    * Foi necessário o enriquecimento dos dados para ter algumas novas features para determinar qual seria o melhor modelo.
            
    * Foi necessário tratar e até mesmo remover registros nulos.
            
    * Foi necessário algumas tratativas para unificar os valores das features e reduzir a quantidade.
    """)

# --- ABA 1: ANÁLISE E PERFORMANCE DO MODELO ---
with tab_analise:
    st.subheader('Análise de dados por ano (2023 e 2024)')
    
    colunas = [
        'classificacao_acidente',
        'tipo_acidente',
        'fase_dia',
        'tipo_pista',
        'tipo_veiculo',
        'tracado_via_unico',
        'final_semana',
        'condicao_tempo',
        'rodovia'
    ]   

    try:

        for c in colunas:
            st.image(f'comparacao_{c}.png', caption=f'Comparação de dados da coluna {c.replace("_", " ")} (2023 vs 2024).')
    except FileNotFoundError:
        st.info("Para exibir gráficos, salve-os como imagens .png na mesma pasta do script.")


# --- ABA 2: GRÁFICO DE ANÁLISES DE % ENTRE 2023 VS 2024 ---    
with tab_distribuicao:
    st.subheader('Comparação de dados por ano (2023 e 2024) em % (proporcional)')
    
    colunas = [
        'classificacao_acidente',
        'tipo_acidente',
        'fase_dia',
        'tipo_pista',
        'tipo_veiculo',
        'tracado_via_unico',
        'final_semana',
        'condicao_tempo',
        'rodovia'
    ]   

    try:

       for c in colunas:
           st.image(f'grafico_{c}_data_drift.png', caption=f'Comparação da distribuição da coluna {c.replace("_", " ")} (2023 vs 2024).')
    except FileNotFoundError:
       st.info("Para exibir gráficos, salve-os como imagens .png na mesma pasta do script.")

with tab_correlacao:
    st.subheader('Gráficos de correlações entre as features')

    st.subheader('Gráfico com one-hot encoding (dummies)')
    st.image(f'grafico_one_hot_encoding.png', caption=f'')

    st.subheader('Gráfico label enconder')
    st.image(f'grafico_label_encoder.png', caption=f'')

with tab_modelo:
    st.subheader('Modelos testados e performance')

    st.subheader('Randon Forest - O MODELO ESCOLHIDO')
    st.info("""
    O modelo Random Forest foi selecionado por oferecer a melhor combinação de estabilidade, desempenho e interpretabilidade. A métrica **Recall** foi o fator decisivo, garantindo maior eficácia na identificação da classe minoritária:\n
    * Recall da Classe Minoritária (SEM VÍTIMAS): 0.62, significativamente 
    superior ao 0.09 do XGBoost.
    * Vantagens Adicionais: Menor *overfitting* e maior facilidade de 
    interpretação do modelo.
    * Acurácia Geral: 0.79, indicando uma boa performance geral.
    """)

    st.image(f'2randonforest.png')

    st.subheader('XGBoost')
    st.image(f'1xgboost.png')

    st.subheader('SVM')
    st.image(f'3svm.png')

    st.subheader('Decision Tree')
    st.image(f'4decisiontree.png')

    st.subheader('LightGBM')
    st.image(f'5lightgbm.png')

    

# --- ABA 2: SIMULADOR DE PREVISÃO ---
with tab_previsao:
    st.header('Simular um Novo Acidente')
    st.markdown("Use os seletores abaixo para descrever um acidente e obter a previsão do modelo.")

    # Criar um formulário para os inputs do utilizador
    with st.form("prediction_form"):
        # Os valores em 'options' devem corresponder exatamente às categorias dos seus dados de treino
        col1, col2, col3 = st.columns(3)
        with col1:
            tipo_acidente = st.selectbox('Tipo de Acidente:', options=['COLISAO TRANSVERSAL', 'QUEDA DE OCUPANTE DE VEICULO',
                                                                        'TOMBAMENTO', 'COLISAO LATERAL MESMO SENTIDO', 'COLISAO TRASEIRA',
                                                                        'COLISAO COM OBJETO', 'CAPOTAMENTO', 'SAIDA DE LEITO CARROCAVEL',
                                                                        'ENGAVETAMENTO', 'ATROPELAMENTO DE ANIMAL',
                                                                        'ATROPELAMENTO DE PEDESTRE', 'INCENDIO',
                                                                        'COLISAO LATERAL SENTIDO OPOSTO', 'COLISAO FRONTAL',
                                                                        'DERRAMAMENTO DE CARGA', 'EVENTOS ATIPICOS'])
            fase_dia = st.selectbox('Fase do Dia:', options=['DIA', 'NOITE'])
            tipo_pista = st.selectbox('Tipo de Pista:', options=['DUPLA', 'SIMPLES', 'MULTIPLA'])
        
        with col2:
            condicao_tempo = st.selectbox('Condição do Tempo:', options=['NORMAL', 'ADVERSO'])
            tipo_veiculo = st.selectbox('Tipo de Veículo:', options=['MOTOCICLETA', 'AUTOMOVEL', 'ONIBUS', 'CAMINHAO', 'BICICLETA','OUTROS'])
            tracado_via_unico = st.selectbox('Traçado da Via:', options=['RETA', 'PONTE', 'CURVA', 'INTERSECAO', 'ACLIVE', 'DECLIVE','RETORNO', 'OBRAS', 'ROTATORIA', 'DESVIO'])
        
        with col3:
            final_semana = st.selectbox('É Fim de Semana?', options=[True, False])
            # Para a rodovia, como são muitas, podemos usar um input de texto ou um selectbox com as principais
            rodovia = st.selectbox('Rodovia (Top 5):', options=['RECIFE-BR-101.0', 'JABOATAO DOS GUARARAPES-BR-101.0',
                                                                'IGARASSU-BR-101.0', 'RECIFE-BR-232.0',
                                                                'JABOATAO DOS GUARARAPES-BR-232.0', 'IPOJUCA-BR-101.0',
                                                                'JABOATAO DOS GUARARAPES-BR-408.0', 'PAULISTA-BR-101.0',
                                                                'CABO DE SANTO AGOSTINHO-BR-101.0', 'ABREU E LIMA-BR-101.0',
                                                                'SAO LOURENCO DA MATA-BR-408.0', 'ITAPISSUMA-BR-101.0',
                                                                'RECIFE-BR-408.0', 'MORENO-BR-232.0', 'RECIFE-BR-235.0', 'OUTROS'])
        
        # Botão para submeter o formulário
        submitted = st.form_submit_button("Fazer Previsão")

        if submitted:
            if modelo_rf is not None:
                # 1. Criar um DataFrame com os inputs do utilizador
                dados_input = {
                    'tipo_acidente': [tipo_acidente],
                    'fase_dia': [fase_dia],
                    'tipo_pista': [tipo_pista],
                    'condicao_tempo': [condicao_tempo],
                    'tipo_veiculo': [tipo_veiculo],
                    'tracado_via_unico': [tracado_via_unico],
                    'final_semana': [final_semana],
                    'rodovia': [rodovia]
                }
                input_df = pd.DataFrame(dados_input)
                
                # 2. Aplicar o One-Hot Encoding
                input_encoded = pd.get_dummies(input_df)
                
                # 3. Alinhar as colunas com as colunas do treino para garantir consistência
                input_final = input_encoded.reindex(columns=colunas_treino, fill_value=0)
                
                # 4. Fazer a previsão
                previsao_encoded = modelo_rf.predict(input_final)
                
                # 5. Descodificar o resultado para texto
                resultado = le_alvo.inverse_transform(previsao_encoded)
                
                # Exibir o resultado com destaque
                st.subheader('Resultado da Previsão:')
                if resultado[0] == 'COM VITIMAS':
                    st.error(f'⚠️ A previsão é: **{resultado[0]}**')
                else:
                    st.success(f'✅ A previsão é: **{resultado[0]}**')
            else:
                st.error("O modelo não pôde ser carregado. Verifique os arquivos.")