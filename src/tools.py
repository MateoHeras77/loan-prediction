import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

def categorizar_tipos_basicos(df, return_visual=True):
    """
    Categoriza las columnas del DataFrame en tipos numéricos y de texto.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        El DataFrame a analizar
    return_visual : bool, default=True
        Si es True, retorna una visualización mejorada. Si es False, retorna el diccionario original.
    
    Returns:
    --------
    pandas.DataFrame, HTML, o dict
        Visualización mejorada o diccionario con claves 'numericos' y 'texto'
    """
    tipos_columnas = {
        'numericos': [],
        'texto': []
    }
    
    for columna in df.columns:
        if pd.api.types.is_numeric_dtype(df[columna]):
            tipos_columnas['numericos'].append(columna)
        else:
            tipos_columnas['texto'].append(columna)
    
    if not return_visual:
        return tipos_columnas
    
    # Crear un DataFrame para visualización
    result = []
    total_cols = len(df.columns)
    
    for tipo, columnas in tipos_columnas.items():
        count = len(columnas)
        percentage = (count / total_cols) * 100
        result.append({
            'Tipo de Dato': tipo.capitalize(),
            'Cantidad': count,
            'Porcentaje': f"{percentage:.1f}%",
            'Columnas': ', '.join(columnas) if columnas else "Ninguna"
        })
    
    result_df = pd.DataFrame(result)
    
    # Generar visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de torta
    ax1.pie(
        result_df['Cantidad'], 
        labels=result_df['Tipo de Dato'], 
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#ff9999']
    )
    ax1.set_title('Distribución de Tipos de Datos Básicos')
    ax1.axis('equal')
    
    # Gráfico de barras
    sns.barplot(x='Tipo de Dato', y='Cantidad', data=result_df, ax=ax2, palette=['#66b3ff', '#ff9999'])
    ax2.set_title('Cantidad de Columnas por Tipo de Dato')
    for i, v in enumerate(result_df['Cantidad']):
        ax2.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    
    # Estilo de tabla para mostrar
    styled_df = result_df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#007bff'), 
                                     ('color', 'white'),
                                     ('font-weight', 'bold'),
                                     ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'left')]},
    ]).set_properties(**{
        'border': '1px solid #ddd',
        'padding': '8px'
    }).hide(axis="index")
    
    display(HTML("<h3>Resumen de Tipos de Datos Básicos</h3>"))
    display(styled_df)
    plt.show()
    
    return result_df

def categorizar_tipos_detallados(df, return_visual=True):


    """
    Categoriza las columnas del DataFrame en tipos de datos detallados.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        El DataFrame a analizar
    return_visual : bool, default=True
        Si es True, retorna una visualización mejorada. Si es False, retorna el diccionario original.
    
    Returns:
    --------
    pandas.DataFrame, HTML, o dict
        Visualización mejorada o diccionario con tipos detallados
    """
    tipos_columnas = {
        # Tipos de texto primero
        'object': [],
        'category': [],
        'string': [],
        'boolean': [],
        # Tipos numéricos después
        'integer': [],
        'float': [],
        # Tipos de fecha/hora
        'datetime': [],
        'timedelta': [],
        # Otros tipos
        'otros': []
    }
    
    for columna in df.columns:
        # Tipos de texto
        if pd.api.types.is_object_dtype(df[columna]):
            tipos_columnas['object'].append(columna)
        elif pd.api.types.is_categorical_dtype(df[columna]):
            tipos_columnas['category'].append(columna)
        elif pd.api.types.is_string_dtype(df[columna]):
            tipos_columnas['string'].append(columna)
        elif pd.api.types.is_bool_dtype(df[columna]):
            tipos_columnas['boolean'].append(columna)
        
        # Tipos numéricos
        elif pd.api.types.is_integer_dtype(df[columna]):
            tipos_columnas['integer'].append(columna)
        elif pd.api.types.is_float_dtype(df[columna]):
            tipos_columnas['float'].append(columna)
        
        # Tipos de fecha/hora
        elif pd.api.types.is_datetime64_dtype(df[columna]):
            tipos_columnas['datetime'].append(columna)
        elif pd.api.types.is_timedelta64_dtype(df[columna]):
            tipos_columnas['timedelta'].append(columna)
        
        # Otros
        else:
            tipos_columnas['otros'].append(columna)
    
    tipos_ordenados = {k: v for k, v in tipos_columnas.items() if v}
    
    if not return_visual:
        return tipos_ordenados
    
    # Crear un DataFrame para visualización
    result = []
    total_cols = len(df.columns)
    
    # Definir categorías principales para agrupar
    categoria_principal = {
        'object': 'Texto',
        'category': 'Texto',
        'string': 'Texto',
        'boolean': 'Texto',
        'integer': 'Numérico',
        'float': 'Numérico',
        'datetime': 'Temporal',
        'timedelta': 'Temporal',
        'otros': 'Otros'
    }
    
    colores = {
        'Texto': '#8dd3c7',
        'Numérico': '#bebada',
        'Temporal': '#fb8072',
        'Otros': '#80b1d3'
    }
    
    for tipo, columnas in tipos_ordenados.items():
        count = len(columnas)
        percentage = (count / total_cols) * 100
        categoria = categoria_principal.get(tipo, 'Otros')
        result.append({
            'Tipo Detallado': tipo,
            'Categoría': categoria,
            'Cantidad': count,
            'Porcentaje': f"{percentage:.1f}%",
            'Columnas': ', '.join(columnas) if columnas else "Ninguna"
        })
    
    result_df = pd.DataFrame(result).sort_values(by=['Categoría', 'Tipo Detallado'])
    
    # Generar visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico de torta
    ax1.pie(
        result_df['Cantidad'], 
        labels=result_df['Tipo Detallado'], 
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('pastel', len(result_df))
    )
    ax1.set_title('Distribución de Tipos de Datos Detallados')
    ax1.axis('equal')
    
    # Gráfico de barras agrupadas
    sns.barplot(x='Tipo Detallado', y='Cantidad', hue='Categoría', data=result_df, ax=ax2)
    ax2.set_title('Cantidad de Columnas por Tipo de Dato Detallado')
    for i, v in enumerate(result_df['Cantidad']):
        ax2.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    
    # Estilo de tabla para mostrar
    styled_df = result_df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#007bff'), 
                                     ('color', 'white'),
                                     ('font-weight', 'bold'),
                                     ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'left')]},
    ]).set_properties(**{
        'border': '1px solid #ddd',
        'padding': '8px'
    }).bar(subset=['Cantidad'], color='#d65f5f').hide(axis="index")
    
    # Aplica colores a la columna de categoría
    def color_categoria(val):
        return f'background-color: {colores.get(val, "#ffffff")}'
    
    styled_df = styled_df.applymap(color_categoria, subset=['Categoría'])
    
    display(HTML("<h3>Resumen de Tipos de Datos Detallados</h3>"))
    display(styled_df)
    plt.show()
    
    return result_df


def crosstab_percentages(df, row_variable, col_variable, normalize='index', margins=True, margins_name="Total"):
    """
    Calculates and returns a crosstab with percentage frequencies.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        row_variable (str): The column name for the rows.
        col_variable (str): The column name for the columns.
        normalize (str, optional): Normalization method ('index', 'columns', 'all', or None). Defaults to 'index'.
        margins (bool, optional): Whether to include margins. Defaults to True.
        margins_name (str, optional): Name of the margins column/row. Defaults to "Total".

    Returns:
        pd.DataFrame: The crosstab with percentage frequencies.
    """

    if normalize:
        crosstab = pd.crosstab(df[row_variable], df[col_variable], normalize=normalize, margins=margins, margins_name=margins_name) * 100
    else:
        crosstab = pd.crosstab(df[row_variable], df[col_variable], margins=margins, margins_name=margins_name)
        if margins:
          if normalize == 'index':
            crosstab = crosstab.div(crosstab.loc[:, margins_name], axis=0) * 100
          elif normalize == 'columns':
            crosstab = crosstab.div(crosstab.loc[margins_name, :], axis=1) * 100
          elif normalize == 'all':
            crosstab = crosstab.div(crosstab.loc[margins_name, margins_name]) * 100
        else:
          if normalize == 'index':
            crosstab = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
          elif normalize == 'columns':
            crosstab = crosstab.div(crosstab.sum(axis=0), axis=1) * 100
          elif normalize == 'all':
            crosstab = crosstab.div(crosstab.sum().sum()) * 100
    return crosstab

# Example usage:
# Row percentages
# row_percentages = crosstab_percentages(df, 'Risk_Flag', 'Car_Ownership', normalize='index')
# print("Row Percentages:\n", row_percentages)

# # Column percentages
# col_percentages = crosstab_percentages(df, 'Risk_Flag', 'Car_Ownership', normalize='columns')
# print("\nColumn Percentages:\n", col_percentages)

# # Overall percentages
# all_percentages = crosstab_percentages(df, 'Risk_Flag', 'Car_Ownership', normalize='all')
# print("\nOverall Percentages:\n", all_percentages)

# #No normalization, manual calculation
# no_norm_row_percentages = crosstab_percentages(df, 'Risk_Flag', 'Car_Ownership', normalize= 'index')
# print("\nNo normalization, manual row Percentages:\n", no_norm_row_percentages)