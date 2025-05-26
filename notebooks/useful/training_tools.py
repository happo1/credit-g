from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

from useful.constants import SEED, N_ITER, TRAINVAL_SPLITS, PALETTE, BLUE, RED, PLOTLY_TEMPLATE, FONT_SIZE

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

class ModelTraining:
    def __init__(self, model_name, model, param_grid, preprocessor, X, y, trainval_splits=TRAINVAL_SPLITS, seed=SEED, pos_label=None, resampler=None):
        self.model_name = model_name
        self.model = model
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.X = X
        self.y = y
        self.cv = StratifiedKFold(n_splits=trainval_splits, shuffle=True, random_state=seed)
        self.pos_label = pos_label
        self.resampler = resampler
        
        steps = [
            ('preprocessor', self.preprocessor)
        ]
        
        if resampler:
            steps.append(('oversampling', self.resampler))
            
        steps.append(('classifier', self.model))
        
        self.pipeline = Pipeline(steps)

        self.results = []
        self.mean_scores = None
        self.best_model = None
        self.best_score = -1

    def _set_train_test(self, X_train, X_test, y_train, y_test):
        # Converte para float
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def _store_result(self, grid):
        y_pred = grid.predict(self.X_test)
        y_pred_str = pd.Series(y_pred)
        y_test_str = self.y_test

        classes = np.unique(y_test_str)

        precision = precision_score(y_test_str, y_pred_str, average=None, labels=classes)
        recall = recall_score(y_test_str, y_pred_str, average=None, labels=classes)
        f1 = f1_score(y_test_str, y_pred_str, average=None, labels=classes)

        for i, cls in enumerate(classes):
            self.results.append({
                'Model': self.model_name,
                'Best Params': grid.best_params_,
                'Class': cls,
                'Precision': round(precision[i], 2),
                'Recall': round(recall[i], 2),
                'F1 Score': round(f1[i], 2)
            })

    def _run_grid(self):
        if isinstance(self.param_grid, list):
            for param in tqdm(self.param_grid, desc=f'{self.model_name}'):
                grid = GridSearchCV(
                    self.pipeline,
                    param,
                    cv=self.cv,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                grid.fit(self.X_train, self.y_train)
                self._store_result(grid)

                if hasattr(grid, 'best_score_') and hasattr(grid, 'best_estimator_'):
                    if self.best_model is None or grid.best_score_ > self.best_score:
                        self.best_score = grid.best_score_
                        self.best_model = grid.best_estimator_
        else:
            grid = GridSearchCV(
                self.pipeline,
                self.param_grid,
                cv=self.cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid.fit(self.X_train, self.y_train)
            self._store_result(grid)

            if hasattr(grid, 'best_score_') and hasattr(grid, 'best_estimator_'):
                if self.best_model is None or grid.best_score_ > self.best_score:
                    self.best_score = grid.best_score_
                    self.best_model = grid.best_estimator_

        if self.best_model is None:
            self.best_model = self.pipeline.fit(self.X_train, self.y_train)

    def _get_results(self):
        return pd.DataFrame(self.results).sort_values(by=['Model', 'Class'])

    def run(self, X_train, X_test, y_train, y_test):
        self._set_train_test(X_train, X_test, y_train, y_test)
        self._run_grid()

        df_results = pd.DataFrame(self.results)
        numeric_cols = df_results.select_dtypes(include=['number'])
        self.mean_scores = round(df_results.groupby(['Model', 'Class'])[numeric_cols.columns].mean(), 2)

        return self._get_results(), self.mean_scores, self.best_model

class ModelMetrics:
    def __init__(self, best_model, X_train, X_test, y_train, y_test, pos_label=None):
        self.best_model = best_model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pos_label = pos_label

    def plot_confusion_matrix(self):
        # 1) Predição
        y_pred = self.best_model.predict(self.X_test)

        # 2) Converter o tipo de y_pred para o mesmo de y_test
        #    - se y_test for um array numpy:
        try:
            y_pred = y_pred.astype(self.y_test.dtype)
        except AttributeError:
            # se y_test for, por exemplo, pandas Series
            y_pred = pd.Series(y_pred).astype(self.y_test.dtype).values

        # 3) Definir a ordem de labels (opcional, mas bom para controlar pos_label)
        if self.pos_label is not None and self.pos_label in self.best_model.classes_:
            labels = [lbl for lbl in self.best_model.classes_ if lbl != self.pos_label] + [self.pos_label]
        else:
            labels = list(self.best_model.classes_)

        # 4) Calcular matriz
        cm = confusion_matrix(self.y_test, y_pred, labels=labels)

        # 5) Plotar
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=np.array([[str(val) for val in row] for row in cm]),
            texttemplate="%{text}",
            textfont={"size": 20}
        ))

        title = f'Matriz de Confusão (pos_label="{self.pos_label}")' if self.pos_label else 'Matriz de Confusão'
        fig.update_layout(
            title=title,
            xaxis_title='Predito',
            yaxis_title='Real',
            template=PLOTLY_TEMPLATE,
            font_size=FONT_SIZE
        )
        fig.update_xaxes(side="bottom")
        return fig
        