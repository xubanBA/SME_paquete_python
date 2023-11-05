import pandas as pd
from .utils import algoritmo_discretizeEW, algoritmo_discretizeEF, entropia, varianza, get_roc, aux_normalizar, aux_estandarizar, info_mutua


######################################
#######     Discretizacion     #######
######################################

def discretizeEW(x, num_bins):
  """
    Discretiza una lista o un DataFrame utilizando el algoritmo Equal Width (igual anchura).

    Utiliza la función algoritmo_discretizeEW() del archivo utils.py para aplicar el algoritmo.
    En esta función se comprueba de que tipo es el argumento recibido para actuar de una forma u otra.
    En cualuier caso, filtrá las variables para escoger solo las numéricas.

    Args:
    x (list or DataFrame): La lista o DataFrame que se va a discretizar.
    num_bins (int): El número intervalos en los que se discretizará.

    Returns:
    Depende de la entrada si es lista o dataframe la salida se guardará en estructuras de datos distintos:

      - Si la entrada es una lista, devuelve una tupla con la discretizacion en forma categorica y la lista con los
                                    puntos de corte.

      - Si la entrada es un dataframe: devuelve un dataframe con los resultados de cada columna. La primera fila muestra
                                      la discretizacion y la segunda los puntos de corte.

    """

  if isinstance(x, list): # si es atributo en forma de lista
    es_numerico = all(isinstance(valor, (int, float)) for valor in x)
    if es_numerico:
      x_discretized, cut_points = algoritmo_discretizeEW(x, num_bins)

      return x_discretized, cut_points

  elif isinstance(x, pd.DataFrame): # si es dataframe
    x = x.select_dtypes(include=['int', 'float']) # numerico

    return x.apply(algoritmo_discretizeEW, num_bins=num_bins)



def discretizeEF(x, num_bins):
  """
    Discretiza una lista o un DataFrame utilizando el algoritmo Equal Frequency (igual frecuencia).

    Utiliza la función algoritmo_discretizeEF() del archivo utils.py para aplicar el algoritmo.
    En esta función se comprueba de que tipo es el argumento recibido para actuar de una forma u otra.
    En cualuier caso, filtrá las variables para escoger solo las numéricas.

    Args:
    x (list or DataFrame): La lista o DataFrame que se va a discretizar.
    num_bins (int): El número intervalos en los que se discretizará.

    Returns:
    Depende de la entrada si es lista o dataframe la salida se guardara en estructuras de datos distintos:

      - Si la entrada es una lista, devuelve una tupla con la discretizacion en forma categorica y la lista con los
                                    puntos de corte.

      - Si la entrada es un dataframe: devuelve un dataframe con los resultados de cada columna. La primera fila muestra
                                      la discretizacion y la segunda los puntos de corte.

    """

  if isinstance(x, list): # si es atributo en forma de lista
    es_numerico = all(isinstance(valor, (int, float)) for valor in x)
    if es_numerico:
      x_discretized, cut_points = algoritmo_discretizeEF(x, num_bins)

      return x_discretized, cut_points

  elif isinstance(x, pd.DataFrame): # si es dataframe
    x = x.select_dtypes(include=['int', 'float']) # numerico

    return x.apply(algoritmo_discretizeEF, num_bins=num_bins) 
  


################################
#######     Métricas     #######
################################

def varianza_dataframe(df):
  """
    Calcula la varianza de las columnas continuas de un DataFrame.

    Utiliza la función varianza() del archivo utils.py para calcular la varianza para cada columna.

    Args:
    df (pd.DataFrame): DataFrame de pandas que contiene columnas con cualquier tipo de datos.
                       La función filtrara el dataframe para escoger solo las columnas continuas.

    Returns:
    pd.Series: Serie de pandas que contiene los valores de entropía para cada columna continua del DataFrame.
               Al hacer uso de la función df.apply() este devuelve el resultado en la estructura Serie de pandas.
               Los índices de la Serie corresponden a los nombres de las columnas del DataFrame original.
  """

  df = df.select_dtypes(include=['float'])
  # es_continua = all(isinstance(valor, int) for valor in df[nombre_columna]) # Otra forma para lo mismo

  return df.apply(varianza)



def entropia_dataframe(df):
  """
    Calcula la entropía de las columnas enteras de un DataFrame.
    Utiliza la función entropia() del archivo utils.py para calcular la entropia para cada columna.

    Args:
    df (pd.DataFrame): DataFrame de pandas que contiene columnas con cualquier tipo de datos.
                       La función filtrara el dataframe para escoger solo las columnas discretas.

    Returns:
    pd.Series: Serie de pandas que contiene los valores de entropía para cada columna discreta del DataFrame.
               Al hacer uso de la función df.apply() este devuelve el resultado en la estructura Serie de pandas.
               Los índices de la Serie corresponden a los nombres de las columnas del DataFrame original.
  """

  df = df.select_dtypes(include=['int'])

  return df.apply(entropia)



def roc_dataframe(df):
  """
    Calcula las curvas ROC para las columnas continuas de un DataFrame utilizando las etiquetas de la última columna
    como valores de clase.
    Utiliza la función get_roc() del archivo utils.py para calcular para cada columna el área debajo de la curva ROC y las listas
    FPR y TPR.

    Args:
    df (DataFrame): El DataFrame que contiene las columnas de cualquier tipo y la última columna debe de ser las etiquetas.
                    La función filtrara el dataframe para escoger solo las columnas continuas.

    Returns:
    pd.Series: Serie de pandas que contiene los valores del área debajo de la curva, la lista TPR y FPR.
               Al hacer uso de la función df.apply() este devuelve el resultado en la estructura Serie de pandas.

    """

  col_etiqueta = df.iloc[:, -1] # Guardar los labels
  df = df.select_dtypes(include=['float']) # Seleccionar las columnas continuas

  return df.apply(get_roc, etiquetas=col_etiqueta)



#######################################################
#######     Normalización y Estandarización     #######
#######################################################

def normalizar(x):
    """
    Normaliza una lista de valores numéricos o un DataFrame con columnas numéricas en el rango [0, 1].
    Utiliza la función aux_normalizar() del archivo utils.py para normalizar la lista o las columnas del dataframe.

    Args:
    x (list or pd.DataFrame): Lista de valores numéricos o DataFrame que se va a normalizar.

    Returns:
    list or pd.DataFrame: Lista normalizada si se proporciona una lista si es numérico, sino devuelve la propia lista.
                          DataFrame con las columnas numéricas normalizadas si se proporciona un DataFrame.
                          Si el tipo de entrada no es compatible, devuelve un mensaje de error.
    """

    if isinstance(x, list):
        es_numerico = all(isinstance(valor, (int, float)) for valor in x) # Verificar si todos los elementos son numéricos
        if es_numerico:
          return aux_normalizar(x)
        else:
          return x

    elif isinstance(x, pd.DataFrame):
        df = x.copy() # Crear copia para que no modifica el dataframe original
        sel = df.select_dtypes(include=['int', 'float']).columns # Seleccionar solo las columnas numéricas
        df[sel] = df[sel].apply(aux_normalizar)
        return df

    else:
        return "No es ni una lista ni un DataFrame"



def estandarizar(x):
    """
    Estandariza una lista de valores numéricos o un DataFrame con columnas numéricas para que tengan media 0 y desviación estándar 1.
    Utiliza la función aux_estandarizar() del archivo utils.py para estandarizar la lista o las columnas del dataframe.

    Args:
    x (list or pd.DataFrame): Lista de valores numéricos o DataFrame que se va a estandarizar.

    Returns:
    list or pd.DataFrame: Lista estandarizada si se proporciona una lista si es numérico, sino devuelve la propia lista.
                          DataFrame con las columnas numéricas estandarizadas si se proporciona un DataFrame.
                          Si el tipo de entrada no es compatible, devuelve un mensaje de error.
    """

    if isinstance(x, list):
        es_numerico = all(isinstance(valor, (int, float)) for valor in x) # Verificar si todos los elementos son numéricos
        if es_numerico:
          return aux_estandarizar(x)
        else:
          return x

    elif isinstance(x, pd.DataFrame):
        df = x.copy() # Crear copia para que no modifica el dataframe original
        sel = df.select_dtypes(include=['int', 'float']).columns # Seleccionar solo las columnas numéricas
        df[sel] = df[sel].apply(aux_estandarizar)
        return df

    else:
        return "No es ni una lista ni un DataFrame"
    



#############################################
#######     Filtrado de variables     #######
#############################################

def aplicar_filtrado(nombre_func, df, umbral):
    """
    Aplica un filtrado a un DataFrame utilizando una función para calcular una métrica específica y un umbral.
    De este modo, dado un dataframe solo se escogen las columnas que SUPEREN el umbral establecido para la función
    indicada. 

    Las funciones disponibles estan establecdos en el diccionario lista_funciones, donde como key se indica la metrica
    que se desea y como valor a la funcion que lo calcula. Por ahora solo se han añadido las metricas varianza y entropia,
    pero si se quiere añadir otras solo hay que indicar en el diccionario.

    Parameters:
    nombre_func (str): Nombre de la función a aplicar. Las funciones válidas son 'varianza' y 'entropia'.
    df (DataFrame): El DataFrame al cual se le aplicará el filtrado.
    umbral (float): El umbral para filtrar los resultados de la función.

    Returns:
    DataFrame: Un nuevo DataFrame que contiene solo las columnas del DataFrame original para las cuales
               el resultado de la función es mayor que el umbral (además de las columnas que no son compatibles con la metrica indicada).
               Si el nombre de la función no es válido, devuelve un mensaje indicando que la función no es válida.
    """

    # Mapeo de nombres de funciones a funciones reales
    lista_funciones = {
        'varianza': varianza_dataframe,
        'entropia': entropia_dataframe
    }

    # Verificar si la función existe en el diccionario
    if nombre_func in lista_funciones:
      df_nuevo = pd.DataFrame()
      res = lista_funciones[nombre_func](df) # Llamada funcion pasando el dataframe
      for i in range(len(res)):
        if res[i] > umbral:
          df_nuevo[res.index[i]] = df[res.index[i]]

      return df_nuevo

    else:
        return "Función no válida"
  


#######################################################
#######     Correlación e Informacion Mutua     #######
#######################################################

def calcular_correlacion(df):
  """
    Calcula la matriz de correlación para las columnas numéricas de un DataFrame.

    Hace uso de la funcion cor() que ofrece pandas.
    Los valores estarán en el rango de -1 a 1, donde 1 indica una correlación positiva perfecta, -1 indica una correlación
    negativa perfecta y 0 indica falta de correlación.

    Parameters:
    df (DataFrame): El DataFrame del cual se calculará la correlación. Las columnas ueden ser de cualquier tipo,
                    pero se escogerán solo aquellas que sean numéricas.

    Returns:
    DataFrame: Un dataframe en forma de matriz de correlación para las columnas numéricas del DataFrame.
               Las columnas y filas mantendran los mismos nombres que el dataframe original.

   """

  df = df.select_dtypes(include=['int', 'float']) # Seleccionar las columnas numéricas

  return df.corr()



def calcular_info_mutua(df):
  """
    Calcula la información mutua entre las columnas categóricas de un DataFrame.
    Utiliza la función info_mutua() del archivo utils.py para calcular la informaciOn mutua entre pares de columnas.

    Los valores estarán en un rango positivo, donde valores más altos indican una mayor dependencia entre las variables correspondientes.

    Parameters:
    df (DataFrame): El DataFrame del cual se calculará la información mutua. Las columnas ueden ser de cualquier tipo,
                    pero se escogerán solo aquellas que sean categoricas.

    Returns:
    DataFrame: Un dataframe en forma de matriz de informacion mutua para las columnas categóricas del DataFrame.
               Las columnas y filas mantendran los mismos nombres que el dataframe original.

    """

  df = df.select_dtypes(include=['category'])

  return info_mutua(df)