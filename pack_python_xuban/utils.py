import pandas as pd
import math

######################################
#######     Discretizacion     #######
######################################

def algoritmo_discretizeEW(x, num_bins):
  """
    Aplica el algoritmo Equal Width (igual anchura) para obtener los puntos de corte y así
    poder discretizar una lista o columna numérica.

    Args:
    x (list o Series): La columna numérica que se va a discretizar.
    num_bins (int): El número intervalos en los que se discretizará.

    Returns:
    pd.Categorical: Una columna categórica que representa la columna discretizada.
    list: Una lista de puntos de corte utilizados para discretizar la columna.
  """

  # Empezar aplicando el algoritmo equal width para determinar los puntos de corte
  minv, maxv = min(x), max(x)
  interv = (maxv - minv) / num_bins
  cut_points = [minv + i*interv for i in range(1, num_bins)]

  # Discretizar tomando cada valor de x para mirar en que intervalo cae
  x_discretized = []
  for val in x:
    for i, punto in enumerate(cut_points):
          if val <= punto:
            x_discretized.append(f"I{i + 1}") # Para devolver estilo: ["I3", "I2", "I1"]
            break # Pasar a analizar la siguente

          if i == len(cut_points) - 1: # Si ha llegado hasta aqui es la última, por lo que está fuera
            x_discretized.append(f"I{num_bins}")

  return(pd.Categorical(x_discretized), cut_points)



def algoritmo_discretizeEF(x, num_bins):
  """
    Aplica el algoritmo Equal Frequency (igual frecuencia) para obtener los puntos de corte y así
    poder discretizar una lista o columna numérica.

    Args:
    x (list o Series): La columna numérica que se va a discretizar.
    num_bins (int): El número intervalos en los que se discretizará.

    Returns:
    pd.Categorical: Una columna categórica que representa la columna discretizada.
    list: Una lista de puntos de corte utilizados para discretizar la columna.
  """

  # Empezar aplicando el algoritmo equal Frequency para determinar los puntos de corte
  interval_size = len(x) // num_bins   # Calcular el tamaño de cada intervalo
  sorted_x = sorted(x)  # Ordenar la lista de entrada
  cut_points = [sorted_x[i * interval_size] for i in range(1, num_bins)] # Sacar los puntos de corte

  # Discretizar tomando cada valor de x para mirar en que intervalo cae
  x_discretized = []
  for val in x:
    for i, punto in enumerate(cut_points):
          if val <= punto:
            x_discretized.append(f"I{i + 1}") # Para devolver estilo: ["I3", "I2", "I1"]
            break # pasar a analizar la siguente

          if i == len(cut_points) - 1: # si ha llegado hasta aqui es la ultima, por lo que esta fuera
            x_discretized.append(f"I{num_bins}")

  return(pd.Categorical(x_discretized), cut_points)




################################
#######     Métricas     #######
################################

def varianza(col):
  """
    Calcula la varianza de una lista cuyos elementos sean continuos. Originalmente esta implementado para utilizar con columnas
    de un dataframe, pero se puede utilizar con cualquier lista.

    Esta pensado para usar esta funcion con apply() de pandas en la función varianza_dataframe() de main_funcs.py,
    por eso recibe Serie de pandas. Pero tambien puede recbir una lista de enteros como entrada.

    Args:
    col (list o Series): Lista de números para los cuales se calculará la varianza.

    Returns:
    float: Valor de la varianza.
  """
  if (len(col) != 0):
    media = sum(col) / len(col) # Calcular la media
    return sum((x - media) ** 2 for x in col) / len(col) # Aplicar la formula de la varianza



def entropia(x):
    """
    Calcula la entropía de una lista cuyos elementos sean discretos.
    Utiliza la fórmula: H(X) = -sum(p(x) * log2(p(x))),  donde p(x) son las probabilidades de cada valor en el conjunto de datos

    Esta pensado para usar esta funcion con apply() de pandas en la función entropia_dataframe() de main_funcs.py,
    por eso recibe Serie de pandas. Pero tambien puede recbir una lista de enteros como entrada.

    Args:
    x (list o Series): Lista de valores para los cuales se calculará la entropía.

    Returns:
    float: Valor de entropía del conjunto de datos si se cumple la condición de que los valores sean discretos.
    """

    # Empezar contando las apariciones de cada elemento
    cont_val = {}
    for val in x:
        cont_val[val] = cont_val.get(val, 0) + 1

    # Calcula las Probabilidades de cada valor
    probs = [cont / len(x) for cont in cont_val.values()]

    # Calcular la entropía dada las probabilidades
    return -sum(p * math.log2(p) if p != 0 else 0 for p in probs)



def get_roc(col, etiquetas):
  """
    Calcula la curva ROC para una columna continua y las etiquetas de clase proporcionadas.
    Esta pensado para usar esta funcion con apply() de pandas en la función roc_dataframe() de main_funcs.py,
    por eso recibe Serie de pandas.

    Args:
    col (Series): La columna continua para la cual se calculará la curva ROC.
    etiquetas (Series): Las etiquetas de clase correspondientes a la columna continua.

    Returns:
    float: El área bajo la curva ROC (AUC).
    list: Una lista de las tasas de verdaderos positivos (TPR) para cada punto de la curva ROC.
    list: Una lista de las tasas de falsos positivos (FPR) para cada punto de la curva ROC.
    """

  # Ordenar el DataFrame
  col = col.sort_values()

  # Inicializar listas para almacenar TPR y FPR
  tpr_list = []
  fpr_list = []

  # Determinar los puntos de la curva ROC (es decir, todos los pares TPR y FPR para cada posible valor de corte)
  for valor_corte in col:
    # Calcular TP, FP, TN, FN para el valor de corte dado
    TP = ( (col >= valor_corte) & (etiquetas==True) ).sum()
    FP = ( (col >= valor_corte) & (etiquetas==False) ).sum()
    TN = ( (col < valor_corte)  & (etiquetas==False) ).sum()
    FN = ( (col < valor_corte)  & (etiquetas==True) ).sum()

    # Calcular TPR y FPR para el valor de corte actual
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    # Guardar TPR y FPR del valor de corte en la lista
    tpr_list.append(TPR)
    fpr_list.append(FPR)

  # Calcular el área bajo la curva ROC (AUC) usando la regla del trapecio
  auc = 0
  for i in range(1, len(tpr_list)):
      auc += 0.5 * (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1])

  return auc, tpr_list, fpr_list



#######################################################
#######     Normalización y Estandarización     #######
#######################################################

def aux_normalizar(x):
  """
    Normaliza una lista de valores numéricos en el rango [0, 1].
    Para normalizar los valores se utiliza la fórmula: (val - min_valor) / (max_valor - min_valor)

    Args:
    x (list o Series): Lista de valores numéricos que se van a normalizar.

    Returns:
    list: Lista normalizada de valores en el rango [0, 1].
          Si (max_valor - min_valor) == 0, devuelve la lista original para no dividir por 0.
  """

  min_valor, max_valor = min(x), max(x) # Calcular el valor mínimo y máximo en la lista
  if (max_valor - min_valor) != 0: 
    return [(val - min_valor) / (max_valor - min_valor) for val in x] # Devolver la lista normalizada
  else:
    return x



def aux_estandarizar(x):
  """
    Estandariza una lista de valores numéricos para que tengan media 0 y desviación estándar 1.
    Para estandarizar se ha utilizado la formula:  (val - media) / des_estandar

    Args:
    x (list o Series): Lista de valores numéricos que se van a estandarizar.

    Returns:
    list: Lista estandarizada de valores con media 0 y desviación estándar 1.
          Si des_estandar==0, devuelve la lista original para no dividir por 0.
  """

  media = sum(x) / len(x)
  des_estandar = (sum((val - media) ** 2 for val in x) / len(x)) ** 0.5
  if des_estandar != 0:
    return[(val - media) / des_estandar for val in x]
  else:
    return x

  


#######################################################
#######     Correlación e Informacion Mutua     #######
#######################################################
  
def entropia_conjunta(x, y):
  """
    Calcula la entropía conjunta de dos listas de valores.
    Función auxiliar para calcular la informacion mutua entre 2 variables.

    Al igual que la funcion entropia() utiliza la fórmula: H(X) = -sum(p(x) * log2(p(x))),  donde ahora p(x)
    son las probabilidades conjuntas de cada valor de una lista con la otra.

    Args:
    x (list): Lista de valores para la primera variable.
    y (list): Lista de valores para la segunda variable.

    Returns:
    float: Valor de entropía conjunta.
  """

  # Calcular las probabilidades conjuntas
  cont_val = {}

  for i in range(len(x)):

    valor_col1 = x[i]
    valor_col2 = y[i]
    par = (valor_col1, valor_col2)
    cont_val[par] = cont_val.get(par, 0) + 1

  # Calcula las Probabilidades Conjuntas de Cada Par (x, y)
  probs_conjuntas = [cont / len(x) for cont in cont_val.values()]

  # Calcular la entropía conjunta dada las probabilidades
  return -sum(p * math.log2(p) if p != 0 else 0 for p in probs_conjuntas)



def info_mutua(df):
  """
    Calcula la información mutua entre todas las columnas de un DataFrame.
    Es una función auxiliar que se usa en la función calcular_info_mutua() donde se pasa como argumento
    un dataframe con la columnas categoricas.

    Para calcular la información mutua entre 2 variables se ha hecho uso de la función:
       información mutua = entropia1 + entropia2 - entropia_conjunta


    Args:
    df (pd.DataFrame): DataFrame de pandas que contiene variables para las cuales se calculará la información mutua.

    Returns:
    pd.DataFrame: DataFrame de pandas que contiene los valores de información mutua para cada par de columnas del DataFrame.
                  Los nombres de las columnas y las filas del DataFrame corresponden a las columnas del DataFrame original.
  """

  mat_mut = []

  # Obtener nombres de las columnas
  columnas = df.columns

  # Procesar todos los pares de columnas
  for i in range(len(columnas)):
    fila = []
    # Calcular entropia 1
    entr1 = entropia(df[columnas[i]])

    for j in range(len(columnas)):
      # Calcular entropia 2
      entr2 = entropia(df[columnas[j]])

      # Calcular entropia conjunta
      entr_conj =  entropia_conjunta(df[columnas[i]], df[columnas[j]])

      # Calcular información muta
      inf_mut = entr1 + entr2 - entr_conj

      # Guardar en la matriz
      fila.append(inf_mut)

    mat_mut.append(fila)

  return pd.DataFrame(mat_mut, columns=columnas, index=columnas)




   