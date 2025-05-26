import pandas as pd
import numpy as np
import pprint

class EDAbasics:
    def __init__(self, data):
        self.data = data
        self.target = None
        self.features = None
        self.numerical_features = None
        self.categorical_features = None
        self.boolean_features = None
        self.heuristic_features = None

    def separate_data(self, target_column):
        """
        Separa os dados em variáveis independentes (features) e alvo (target).
        """
        if target_column not in self.data.columns:
            raise ValueError(f"'{target_column}' não está presente nas colunas do DataFrame.")
        
        self.features = self.data.drop(columns=[target_column])
        self.target = self.data[target_column]

    def features_categories(self):
        """
        Separa as features em categorias:
        - Numéricas
        - Categóricas
        - Booleanas
        - Heurísticas (inteiros com poucos valores únicos)
        """
        if self.features is None:
            self.features = self.data.copy()

        df = self.features

        # Numéricas
        self.numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Categóricas
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()

        # Booleanas
        self.boolean_features = df.select_dtypes(include=[bool]).columns.tolist()

        # Heurísticas (10 ou menos valores únicos e inteiros)
        self.heuristic_features = [
            col for col in df.columns
            if df[col].nunique() <= 10 and pd.api.types.is_integer_dtype(df[col])
        ]

        # Remove heurísticas das numéricas
        self.numerical_features = [
            col for col in self.numerical_features
            if col not in self.heuristic_features
        ]

        return (
            self.numerical_features,
            self.categorical_features,
            self.boolean_features,
            self.heuristic_features
        )

    def summary(self):
        """
        Exibe um resumo das features identificadas.
        """
        print("Numerical Features:", self.numerical_features)
        print("Categorical Features:", self.categorical_features)
        print("Boolean Features:", self.boolean_features)
        print("Heuristic Features:", self.heuristic_features)
        
# ----------------------------------------------------------------------------------------------------
    
class IQR_Calc:
    def __init__(self, df, numerical_columns):
        self.df = df
        self.numerical_columns = numerical_columns
        self.results = {}

    def execute(self):
        for col in self.numerical_columns:
            data = self.df[col]

            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr

            outliers = data[(data < limite_inferior) | (data > limite_superior)]

            self.results[col] = {
            'outliers': outliers
            }

    def summary(self):
        print(f"Calculando outliers com IQR")
        print("-" * 30)
        for col, result in self.results.items():
            print(f"Coluna: {col}")
            print(f"Outliers: {len(result['outliers'])}")
            print("-" * 30)
            
    # DF somente com os outliers        
    def get_outliers(self, dataset_new_name):
        outlier_indices = set()
        for col, result in self.results.items():
            outlier_indices.update(result['outliers'].index)

        # Retorna um DataFrame apenas com as linhas que contêm outliers
        dataset_new_name = self.df.loc[self.df.index.isin(outlier_indices)]
        print(f"Total de outliers encontrados {len(outlier_indices)}")
        print(f"Total de linhas no DataFrame com outliers {len(dataset_new_name)}")
        return dataset_new_name.reset_index(drop=True)
        
        
# ----------------------------------------------------------------------------------------------------
    
class ZScore_Calc:
    def __init__(self, df, numerical_columns):
        self.df = df
        self.numerical_columns = numerical_columns
        self.results = {}

    def execute(self):
        for col in self.numerical_columns:
            data = self.df[col]

            mean = data.mean()
            std_dev = data.std()

            z_scores = (data - mean) / std_dev

            outliers = data[abs(z_scores) > 3]

            self.results[col] = {
                'outliers': outliers
            }

    def summary(self):
        print(f"Calculando outliers com Z-Score")
        print("-" * 30)
        for col, result in self.results.items():
            print(f"Coluna: {col}")
            print(f"Outliers: {len(result['outliers'])}")
            print("-" * 30)
    
    def get_outliers(self, dataset_new_name):
        outlier_indices = set()
        for col, result in self.results.items():
            outlier_indices.update(result['outliers'].index)

        # Retorna um DataFrame igual ao original, mas sem essas linhas
        dataset_new_name = self.df.drop(index=sorted(outlier_indices))
        print(f"Total de linhas a serem removidas {len(outlier_indices)}")
        print(f"Total de linhas a serem mantidas {len(dataset_new_name)}")
        return dataset_new_name.reset_index(drop=True)

# ----------------------------------------------------------------------------------------------------