#Funções auxiliares utilizadas no minicurso "Consolidando sua carteira com Python"
#X Jornada de Ciência e Tecnologia do IFMG Campus Formiga
#Ministrante: Maísa Kely de Melo
#Data: 21 de outubro de 2021

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def get_information(asset_ticker,df_invest):
    ticket = yf.Ticker(str(asset_ticker)+'.SA')
    data_inicial = str(df_invest[df_invest['nome investimento'] == str(asset_ticker)]['data'].iloc[0])[:10]
    data_final = str(pd.Timestamp.today())[:10]
    df = ticket.history(interval = '1d',start=data_inicial, end = data_final, rounding=True)
    return df

def get_inverse_cumprod(X):
    return (np.prod(X) / np.cumprod(X)) * X

def get_data_split_inplit(asset_ticker,df_invest):
    df = get_information(asset_ticker,df_invest)
    datas_split_inplit = pd.DataFrame(df[df['Stock Splits'] != 0]['Stock Splits'], index = df[df['Stock Splits'] != 0].index)
    datas_split_inplit['inverse_cumprod'] = get_inverse_cumprod(datas_split_inplit['Stock Splits'])
    return datas_split_inplit

def get_adj_df_asset(df_asset,datas_split_inplit):
    for i,data in enumerate(df_asset['data']):
        posicao = np.where(data < datas_split_inplit.index)[0] #pega a posição da primeira data cuja data de investimento é menor
        if len(posicao) == 0:
            df_asset['quantidade'].iloc[i] = df_asset['quantidade original'].iloc[i]
            df_asset['valor'].iloc[i] = df_asset['valor original'].iloc[i]
        else:
            df_asset['quantidade'].iloc[i] = df_asset['quantidade original'].iloc[i] * datas_split_inplit['inverse_cumprod'][posicao[0]]
            df_asset['valor'].iloc[i] = df_asset['valor original'].iloc[i] / datas_split_inplit['inverse_cumprod'][posicao[0]]
    df_asset['aporte compara'] = df_asset['quantidade']*df_asset['valor']
    return df_asset

def principal_desdobramentos(df_invest):
    df_invest.sort_values(by=['data'], inplace=True)
    for asset in df_invest['nome investimento'].unique():
        datas_split_inplit = get_data_split_inplit(asset,df_invest)
        df_asset = df_invest[df_invest['nome investimento'] == asset]
        df_asset = get_adj_df_asset(df_asset,datas_split_inplit)
        df_invest[df_invest['nome investimento'] == asset] = df_asset
    return df_invest

def get_prices(df_investimento):
    # pega cotações dos ativos
    data_inicial = str(df_investimento['data'].iloc[0])[:10]
    prices = yf.download(tickers=(df_investimento['nome investimento'] + '.SA').tolist(),
                                        start=data_inicial, rounding=True)['Adj Close']
    prices.columns = prices.columns.str.rstrip('.SA')  # tira o .SA do nome do ativo
    days_to_drop = prices.index[prices.isnull().all(axis=1)]
    prices.drop(days_to_drop,inplace = True)
    return prices

def preenche_cotas(carteira,aportes):
    for i,data in enumerate(aportes.index):
        if i == 0:
            carteira.at[data,'vl_cota'] = 1
            carteira.at[data,'qtd_cotas'] = carteira.loc[data]['saldo'].copy()
        else:
            if aportes[data] != 0:
                carteira.at[data,'qtd_cotas'] = carteira.iloc[i-1]['qtd_cotas'] +(aportes[data] / carteira.iloc[i-1]['vl_cota'])
                carteira.at[data,'vl_cota'] = carteira.iloc[i]['saldo']/carteira.at[data,'qtd_cotas']
                carteira.at[data,'retorno'] = (carteira.iloc[i]['vl_cota']/carteira.iloc[i-1]['vl_cota'])-1
            else:
                carteira.at[data,'qtd_cotas'] = carteira.iloc[i-1]['qtd_cotas']
                carteira.at[data,'vl_cota'] = carteira.iloc[i]['saldo']/carteira.at[data,'qtd_cotas']
                carteira.at[data,'retorno'] = (carteira.iloc[i]['vl_cota']/carteira.iloc[i-1]['vl_cota'])-1
    return carteira

def consolida_portfolio(df_investimento):
    prices = get_prices(df_investimento)
    df_investimento['data'] = pd.to_datetime(df_investimento['data'])
    df_investimento = df_investimento.sort_values(by='data', ascending=True)
    trade_quant = pd.pivot_table(df_investimento, values = 'quantidade', index = ['data'], columns = df_investimento['nome investimento'],
                                 aggfunc=np.sum,fill_value=0)
    trade_price = pd.pivot_table(df_investimento, values = 'valor', index = ['data'],
                                 columns = df_investimento['nome investimento'],fill_value=0)
    trades = trade_quant.reindex(index = prices.index)
    trades.fillna(value=0,inplace=True)
    aportes = (trades*trade_price).sum(axis = 1)
    posicao = trades.cumsum()
    carteira = posicao * prices
    carteira['saldo'] = carteira.sum(axis=1)
    carteira = preenche_cotas(carteira,aportes)
    return carteira

def ibovespa(df_investimento):
    data_inicial = str(df_investimento['data'].iloc[0])[:10]
    # Define retorno acumulado do Ibovespa
    ibovespa = yf.download(tickers=['^BVSP'], start=data_inicial, rounding=True)['Adj Close']
    ibov_ret = ibovespa.pct_change()
    cumulative_ret_ibov = (1 + ibov_ret).cumprod() - 1
    return ibov_ret, cumulative_ret_ibov

def cdi(portfolio):
    # Obtendo Dados através da API do Banco Central do Brasil
    # https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries
    def consulta_bc(codigo_bcb):
        url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)
        df = pd.read_json(url)
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df.set_index('data', inplace=True)
        return df

    cdi = consulta_bc(12)  # % p.d.
    cdi = cdi / 100
    cdi_datas_corretas = cdi.reindex(portfolio.index)
    cdi_acumulado = (1 + cdi_datas_corretas).cumprod() - 1
    return cdi_datas_corretas, cdi_acumulado

def get_composicao_atual(carteira):
    composicao_atual = carteira[carteira.columns[:-4]].iloc[-1]
    composicao_atual.drop(composicao_atual.index[composicao_atual == 0], inplace = True)
    composicao_atual.dropna(inplace = True)
    return composicao_atual

def get_composicao_carteiras(carteira_renda_variavel, carteira_tesouro, carteira_renda_fixa,carteira_fundos):
    composicao_renda_variável = get_composicao_atual(carteira_renda_variavel)
    composicao_renda_tesouro = get_composicao_atual(carteira_tesouro)
    composicao_renda_fixa = get_composicao_atual(carteira_renda_fixa)
    composicao_fundos = get_composicao_atual(carteira_fundos)
    return composicao_renda_variável, composicao_renda_tesouro, composicao_renda_fixa,composicao_fundos

def get_grafico_composicao(portfolio):
    composicao_atual = get_composicao_atual(portfolio)
    import plotly.graph_objects as go
    labels = composicao_atual.index.tolist()
    values = composicao_atual.values.tolist()
    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, textinfo='label+percent')])
    fig.update(layout_showlegend=False)
    fig.update_layout(
        title={'text': "Composição {}".format('Renda Variável'),
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig.show()
    return

def grafico_rentabilidade_acumulada(retorno_acumulado_ibovespa, retorno_acumulado_cdi,retorno_acumulado_portfolio):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.plot(retorno_acumulado_ibovespa, LineWidth=3, label="Ibovespa")
    ax1.plot(retorno_acumulado_cdi, LineWidth=4, label="CDI")
    ax1.plot(retorno_acumulado_portfolio, LineWidth=4, label="Portfólio")
    ax1.set_xlabel('Datas', fontsize=20)
    legend = plt.legend(loc='upper left', shadow=False, fontsize=18, ncol=1)
    ax1.set_ylabel("Retorno Acumulado", fontsize=20)
    ax1.set_title("Comparação dos retornos acumulados", fontsize=20)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=20)
    return

def cria_informacoes_ativo(df_ativo,ativo,prices):
   #Cria o dataframe com as informações base
    df_carteira = pd.DataFrame({
        'transação' : df_ativo[df_ativo['nome investimento']==str(ativo)]['quantidade'].values.tolist(),
        'valor' : df_ativo[df_ativo['nome investimento']==str(ativo)]['valor'].values.tolist(),
        'preço médio': len(df_ativo[df_ativo['nome investimento']==str(ativo)]['valor'].values.tolist())*[0],
        'saldo carteira': len(df_ativo[df_ativo['nome investimento']==str(ativo)]['valor'].values.tolist())*[0],
    },index = df_ativo[df_ativo['nome investimento']==str(ativo)]['data'])
    #Cria colunas com as informações demandadas
    ########################################################
    df_carteira['total transação'] = df_carteira['transação']*df_carteira['valor']
    #######################################################
    df_carteira['quantidade'] = df_carteira['transação'].cumsum()
    #######################################################
    aux_total_investido = df_carteira[df_carteira['transação']>0]['transação']*df_carteira[df_carteira['transação']>0]['valor']
    df_carteira['total investido'] = aux_total_investido.cumsum()
    ########################################################
    aux_total_vendas = df_carteira[df_carteira['transação']<0]['transação']*df_carteira[df_carteira['transação']<0]['valor']
    df_carteira['total vendas'] = abs(aux_total_vendas).cumsum()
    #Preenche parte referente ao preço médio
    df_carteira['preço médio'].iloc[0] = df_carteira['total investido'].iloc[0]/df_carteira['quantidade'].iloc[0]
    df_carteira['saldo carteira'].iloc[0] = df_carteira['total investido'].iloc[0]

    for data in range(len(df_carteira.index)-1):
        if df_carteira['quantidade'].iloc[data+1]<df_carteira['quantidade'].iloc[data]:
            df_carteira['preço médio'].iloc[data+1] = df_carteira['preço médio'].iloc[data]
            df_carteira['saldo carteira'].iloc[data+1] = df_carteira['quantidade'].iloc[data+1]*df_carteira['preço médio'].iloc[data]
        else:
            saldo = df_carteira['total transação'].iloc[data+1] + df_carteira['saldo carteira'].iloc[data]
            df_carteira['saldo carteira'].iloc[data+1] = saldo
            df_carteira['preço médio'].iloc[data+1] = saldo/df_carteira['quantidade'].iloc[data+1]

    df_carteira['lucro transação'] = (df_carteira[df_carteira['transação'] <0]['preço médio']-df_carteira[df_carteira['transação'] <0]['valor'])*df_carteira[df_carteira['transação'] <0]['transação']
    df_carteira['lucro acumulado'] = df_carteira['lucro transação'].cumsum()
    return df_carteira,prices[str(ativo)][-1]

def get_preco_medio(prices, df_investimento):
    relatorio_portfolio = pd.DataFrame([], columns = ['Quantidade','Preço médio','Preço atual','Posição atual'],
                                       index = df_investimento['nome investimento'].unique().tolist())
    for ativo in relatorio_portfolio.index:
        df_carteira,preco_atual = cria_informacoes_ativo(df_investimento,ativo,prices)
        relatorio_portfolio.loc[ativo]['Quantidade'] = df_carteira['quantidade'][-1]
        relatorio_portfolio.loc[ativo]['Preço médio'] = round(df_carteira['preço médio'][-1],2)
        relatorio_portfolio.loc[ativo]['Preço atual'] = preco_atual
        relatorio_portfolio.loc[ativo]['Posição atual'] = preco_atual*df_carteira['quantidade'][-1]
    relatorio_portfolio.fillna(value=0,inplace=True)
    return relatorio_portfolio

def grafico_rentabilidade_acumulada_acoes(retornos_diarios_acumulados):
    df_ultimo_dia = pd.DataFrame(retornos_diarios_acumulados.iloc[-1]).T
    df_positivo = df_ultimo_dia[df_ultimo_dia > 0]
    df_acoes_positivas = df_positivo.loc[:, ~df_positivo.isnull().any()]
    df_acoes_negativas = df_positivo.loc[:, df_positivo.isnull().any()]
    lista1 = df_acoes_positivas.columns.values.tolist()
    lista2 = df_acoes_negativas.columns.values.tolist()
    retornos_diarios_acumulados[lista1].plot(fontsize=16,
                                             figsize=(15, 5), alpha=0.9, colormap='Blues',
                                             legend=True).set(xlabel='Dias', ylabel='Retorno Acumulado')
    retornos_diarios_acumulados[lista2].plot(fontsize=16,
                                             figsize=(15, 5), alpha=0.9, colormap='Reds',
                                             legend=True).set(xlabel='Dias', ylabel='Retorno Acumulado')

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    return
