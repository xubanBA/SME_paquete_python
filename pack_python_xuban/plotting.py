import matplotlib.pyplot as plt
from  .main_funcs import calcular_correlacion, calcular_info_mutua



def visualizar_matriz_correlacion(datos):
    """
    Visualiza la matriz de correlación para un dataframe.
    Se hace uso de función calcular_correlacion() para obtener los datos y asi poder realizar el plot.

    Args:
    datos (pandas.DataFrame): Un DataFrame que contiene las variables para las cuales se calculará la correlacion.
    """

    # Calcula la matriz de correlaciones
    matriz_correlacion = calcular_correlacion(datos)

    # Configura el tamaño de la figura
    plt.figure(figsize=(10, 8))

    # Crea un mapa de calor para visualizar la matriz de correlaciones
    plt.imshow(matriz_correlacion, interpolation='nearest', cmap='coolwarm')
    plt.colorbar()

    # Etiquetas de los ejes
    num_variables = len(matriz_correlacion.columns)
    plt.xticks(range(num_variables), matriz_correlacion.columns, rotation=90)
    plt.yticks(range(num_variables), matriz_correlacion.columns)

    # Muestra el valor de correlación en cada celda
    for i in range(num_variables):
        for j in range(num_variables):
          plt.text(i, j, round(matriz_correlacion.iloc[i, j], 2), ha='center', va='center', color='black')

    # Título y etiquetas de los ejes
    plt.title('Matriz de Correlación')
    plt.xlabel('Variables')
    plt.ylabel('Variables')

    # Muestra la matriz de correlaciones
    plt.show()



def visualizar_infmutua(datos):
    """
    Visualiza la matriz de información mutua para un dataframe.
    Se hace uso de función calcular_info_mutua() para obtener los datos y asi poder realizar el plot.

    Args:
    datos (pandas.DataFrame): Un DataFrame que contiene las variables para las cuales se calculará la información mutua.
    """

    # Calcular la matriz de informacion mutua
    matriz_info = calcular_info_mutua(datos)

    # Configura el tamaño de la figura
    plt.figure(figsize=(10, 8))

    # Crea un mapa de calor para visualizar la matriz de correlaciones
    plt.imshow(matriz_info, interpolation='nearest', cmap='coolwarm')
    plt.colorbar()

    # Etiquetas de los ejes
    num_variables = len(matriz_info.columns)
    plt.xticks(range(num_variables), matriz_info.columns, rotation=90)
    plt.yticks(range(num_variables), matriz_info.columns)

    # Muestra el valor de correlación en cada celda
    for i in range(num_variables):
        for j in range(num_variables):
            plt.text(i, j, round(matriz_info.iloc[i, j], 2), ha='center', va='center', color='black')

    # Título y etiquetas de los ejes
    plt.title('Matriz de Informacion Mutua')
    plt.xlabel('Variables')
    plt.ylabel('Variables')

    # Muestra la matriz de correlaciones
    plt.show()


def visualizar_roc(fpr_list, tpr_list):
  """
    Visualiza una curva AUC-ROC utilizando las tasas de falsos positivos (FPR)
    y verdaderos positivos (TPR) proporcionadas.

    Args:
    - fpr_list (list): Lista de tasas de falsos positivos.
    - tpr_list (list): Lista de tasas de verdaderos positivos.
    Los dos argumentos deben de tener la misma longitud
  """

  plt.figure(figsize=(8, 6))
  plt.plot(fpr_list, tpr_list, color='blue', lw=2, label='Curva ROC')
  plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Línea de referencia')
  plt.xlabel('Tasa de Falsos Positivos (FPR)')
  plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
  plt.title('Curva ROC')
  plt.legend()
  plt.grid(True)
  plt.show()