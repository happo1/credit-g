# Importando bibliotecas essenciais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import set_config
import warnings
from useful.constants import SEED
import os

# Função que configura todas as bibliotecas
def set_default_config():
    warnings.simplefilter('ignore'); os.environ['PYTHONWARNINGS'] = 'ignore'
    
    np.random.seed(SEED)
    # Configurações padrão do Pandas
    pd.set_option('display.max_columns', None)      # Exibe todas as colunas
    pd.set_option('display.width', None)            # Não quebra a linha horizontalmente
    pd.set_option('display.max_colwidth', None)     # Mostra todo o conteúdo das células
    pd.set_option('display.max_rows', 20)           # Número de linhas visíveis
    pd.set_option('display.min_rows', 10)           # Número de linhas visíveis
    pd.set_option('display.float_format', '{:.2f}'.format)  # Formato de float
    pd.set_option('mode.chained_assignment', None)  # Desativa o warning de atribuições
    pd.set_option('display.max_colwidth', None)  # Exibe conteúdo completo das colunas
    pd.set_option('display.max_seq_item', None)  # Exibe todas as sequências em listas/dicionários

    # Configurações do Matplotlib
    plt.rcParams.update({
        'figure.figsize': (10, 6),  # Tamanho padrão dos gráficos
        'axes.grid': True,  # Habilita grid nos gráficos
        'axes.titlesize': 14,  # Tamanho do título do gráfico
        'axes.labelsize': 12  # Tamanho dos rótulos dos eixos
    })

    # Configurações do Seaborn
    sns.set_theme(style="darkgrid", palette="muted")
    sns.set_context("notebook")  # Contexto dos gráficos

    # Configurações do NumPy
    np.set_printoptions(precision=2, suppress=True)  # Exibe com 2 casas decimais e sem notação científica

    # Configurações do SciPy
    np.set_printoptions(precision=4)  # Aumenta a precisão para 4 casas decimais

    # Configurações do Scikit-learn
    set_config(display='diagram')  # Exibe diagramas de estimadores no scikit-learn

    # Configurações para warnings
    warnings.filterwarnings('ignore')  # Ignora warnings desnecessários
