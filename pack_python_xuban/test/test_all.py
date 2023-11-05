import pandas as pd
from pack_python_xuban import main_funcs as mf
from pack_python_xuban import plotting as plo
    

## Inicializar los datos para hacer pruebas

# Lista
x = [1.6, 1, 5.1, 6, 3.7]

# Dataframe
data = {
    'Col1': [1.0, 1.0, 2.2, 3.1, 5.5],
    'Col2': [1, 0, 1, 2, 3],
    'Col3': [0, 0, 1, 2, 2],
    'Col4': [1.0, 2.2, 2.2, 5.5, 1.0],
    'Col5': pd.Categorical([2, 2, 3.1, 3.1, 3.1]),
    'Col6': pd.Categorical([3, 2, 2, 4, 5]),
    'Col7': pd.Categorical(['Blanco', 'Negro', 'Negro', 'Blanco','Rojo']),
    'Col8': pd.Categorical(['Rojo', 'Verde', 'Azul', 'Blanco', 'Blanco']),
    'Col9': ['a', 'b', 'c', 'd', 'e'],
    'Col10': [True, False, True, False, True]
}
df = pd.DataFrame(data)


def test_discretitacion():
    # Número de intervalos
    num_bins = 3

    # Prueba con lista
    mf.discretizeEW(x, num_bins) # Equal Width
    mf.discretizeEF(x, num_bins) # Equal Frecuancy
    
    # Prueba con dataframe
    mf.discretizeEW(df, num_bins) # Equal Width
    mf.discretizeEF(df, num_bins) # Equal Frecuancy

def test_metricas():
    mf.varianza_dataframe(df) # Varianza con dataframe
    mf.entropia_dataframe(df) # Entropia con dataframe
    mf.roc_dataframe(df) # Área roc con dataframe

def test_normalizacio_estandarizacion():
    mf.normalizar(x) # Normalizar con lista
    mf.estandarizar(x) # Estandarizar con lista

    mf.normalizar(df) # Normalizar con dataframe
    mf.estandarizar(df) # Estandarizar con dataframe

def test_filtrado():
    mf.aplicar_filtrado('varianza', df, 0.5) # Filtrar pasando una funcion y dataframe

def test_correlacio_informacio_mutua():
    mf.calcular_correlacion(df) # Correlacion con un dataframe
    mf.calcular_info_mutua(df)  # Información mutua con un dataframe

def test_visualizar():

    plo.visualizar_matriz_correlacion(df) # Visualizar matriz correlaicón 
    plo.visualizar_infmutua(df) # Visualizar matriz información mutua 

    fpr_list = [0, 0.1, 0.2, 0.5, 0.8, 1]
    tpr_list = [0, 0.15, 0.45, 0.8, 0.9, 1]
    plo.visualizar_roc(fpr_list, tpr_list) # Visualizar ROC
    