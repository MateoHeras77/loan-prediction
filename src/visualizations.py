import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import chi2_contingency


# Function to plot the distribution of a binary variable

def plot_binary_distribution(df, column_name):
    sns.countplot(x=column_name, data=df)
    plt.title(f'Distribution of {column_name}')
    plt.show()

# Function to plot the correlation matrix

def plot_correlation_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

# Function to plot the distribution of all binary variables

def plot_all_binary_distributions(df):
    binary_columns = df.columns[df.nunique() == 2]
    for column in binary_columns:
        plot_binary_distribution(df, column)

# Function to create subplots for binary variable distributions
def plot_binary_subplots(df, figsize=(15, 10), palette=None, title=None, value_labels=None):
    """
    Create enhanced visualizations for binary variables in a dataframe.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing binary variables
    figsize : tuple, optional
        Figure size (width, height) in inches
    palette : list or dict, optional
        Custom color palette for the plots
    title : str, optional
        Main title for the entire figure
    value_labels : dict, optional
        Dictionary to map binary values to custom labels, e.g. {0: 'No', 1: 'Yes'}
    """
    
    # Find binary columns (columns with exactly 2 unique values)
    binary_columns = df.columns[df.nunique() == 2].tolist()
    
    if not binary_columns:
        print("No binary variables found in the dataset.")
        return
    
    # Set default colors if not provided
    if palette is None:
        palette = ['#3498db', '#e74c3c']  # Blue and red
    
    # Default value labels if not provided
    if value_labels is None:
        value_labels = {0: '0', 1: '1'}
    
    # Set up the grid layout
    n_cols = 3
    n_rows = (len(binary_columns) + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Set a main title if provided
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        # Adjust top margin if title is provided
        top_margin = 0.90
    else:
        top_margin = 0.95
    
    # Create GridSpec to have better control over spacing
    gs = fig.add_gridspec(n_rows, n_cols, top=top_margin, bottom=0.05, 
                         left=0.05, right=0.95, hspace=0.4, wspace=0.3)
    
    # Create a shared x-axis range for all plots
    x_range = [-0.5, 1.5]  # For binary variables (0 and 1)
    
    # Store the maximum count to standardize y-axes
    max_count = 0
    for col in binary_columns:
        count = df[col].value_counts().max()
        if count > max_count:
            max_count = count
    
    # Create subplots
    for i, col in enumerate(binary_columns):
        if i < n_rows * n_cols:
            row_idx = i // n_cols
            col_idx = i % n_cols
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Get value counts and percentages
            value_counts = df[col].value_counts().sort_index()
            total = len(df[col])
            percentages = 100 * value_counts / total
            
            # Create the bar plot with custom colors
            bars = sns.barplot(x=value_counts.index, y=value_counts.values, 
                              palette=palette, ax=ax)
            
            # Format the plot
            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Count', fontsize=10)
            
            # Set x-axis labels using provided value_labels
            ax.set_xticks([0, 1])
            ax.set_xticklabels([value_labels[0], value_labels[1]])
            
            # Set x-axis limits for consistency across all plots
            ax.set_xlim(x_range)
            
            # Set y-axis limit to be the same for all plots
            y_max = max_count * 1.2  # Add 20% padding
            ax.set_ylim(0, y_max)
            
            # Format y-axis with comma for thousands
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
            
            # Add count and percentage labels inside bars
            for j, (bar, percentage) in enumerate(zip(ax.patches, percentages)):
                height = bar.get_height()
                count = value_counts.iloc[j]
                
                # Position the text in the middle of the bar
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 2,  # Center in the bar
                    f'{count:,}\n({percentage:.1f}%)',
                    ha='center', 
                    va='center',
                    fontsize=10, 
                    fontweight='bold',
                    color='white'  # White text for better visibility on colored bars
                )
            
            # Add gridlines for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Remove top and right spines for cleaner look
            sns.despine(ax=ax)
    
    # Hide any unused subplots
    for j in range(len(binary_columns), n_rows * n_cols):
        row_idx = j // n_cols
        col_idx = j % n_cols
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.set_visible(False)
    
    plt.tight_layout()
    return fig

# Function to create pie charts for binary variables
def plot_binary_pie_subplots(df, figsize=(15, 10)):
    binary_columns = df.columns[df.nunique() == 2].tolist()
    
    if not binary_columns:
        print("No binary variables found in the dataset.")
        return
    
    n_cols = 3
    n_rows = (len(binary_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(binary_columns):
        if i < len(axes):
            # Calculate value counts and percentages
            value_counts = df[col].value_counts()
            labels = value_counts.index.tolist()
            sizes = value_counts.values
            
            # Pie chart
            axes[i].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            axes[i].set_title(f'Distribution of {col}')
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Function to create heatmap of contingency tables between binary variables
def plot_binary_contingency_heatmap(df, figsize=(12, 10)):
    binary_columns = df.columns[df.nunique() == 2].tolist()
    
    if len(binary_columns) < 2:
        print("Need at least 2 binary variables for contingency analysis.")
        return
    
    n = len(binary_columns)
    fig, axes = plt.subplots(n, n, figsize=figsize)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Create contingency table
                contingency = pd.crosstab(df[binary_columns[i]], df[binary_columns[j]], 
                                          normalize='all') * 100
                
                # Plot heatmap
                sns.heatmap(contingency, annot=True, fmt='.1f', cmap='Blues', ax=axes[i, j])
                axes[i, j].set_title(f'{binary_columns[i]} vs {binary_columns[j]}')
                axes[i, j].set_xlabel('')
                axes[i, j].set_ylabel('')
            else:
                # On diagonal, show variable name
                axes[i, j].text(0.5, 0.5, binary_columns[i], 
                              horizontalalignment='center',
                              verticalalignment='center',
                              fontsize=12, fontweight='bold')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].spines['top'].set_visible(False)
                axes[i, j].spines['right'].set_visible(False)
                axes[i, j].spines['bottom'].set_visible(False)
                axes[i, j].spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Function to create stacked bar charts for binary variables
def plot_binary_stacked_bars(df, figsize=(15, 10)):
    binary_columns = df.columns[df.nunique() == 2].tolist()
    
    if len(binary_columns) < 2:
        print("Need at least 2 binary variables for stacked bar analysis.")
        return
    
    n = len(binary_columns)
    fig, axes = plt.subplots(n-1, 1, figsize=figsize)
    
    if n == 2:
        axes = [axes]  # Make axes iterable if only one subplot
    
    for i in range(n-1):
        # Create crosstab
        ct = pd.crosstab(df[binary_columns[i]], df[binary_columns[i+1]], normalize='index') * 100
        
        # Plot stacked bars
        ct.plot(kind='bar', stacked=True, ax=axes[i], colormap='viridis')
        axes[i].set_title(f'{binary_columns[i]} vs {binary_columns[i+1]}')
        axes[i].set_ylabel('Percentage')
        axes[i].legend(title=binary_columns[i+1])
        
        # Add percentage annotations
        for p in axes[i].patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            if height > 5:  # Only annotate if segment is large enough
                axes[i].text(x + width/2, y + height/2, f'{height:.1f}%', 
                            ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

# Function to create a complete dashboard of binary visualizations
def binary_dashboard(df, figsize=(18, 12), association_method='phi'):
    """
    Crea un dashboard con múltiples visualizaciones para variables binarias
    
    Parameters:
    df : pandas DataFrame
    figsize : tuple, tamaño de la figura
    association_method : str, método para calcular asociación entre variables binarias
    """
    binary_columns = df.columns[df.nunique() == 2].tolist()
    
    if not binary_columns:
        print("No se encontraron variables binarias en el dataset.")
        return
    
    # Crear un grid para organizar los subgráficos
    fig = plt.figure(figsize=figsize)
    
    # Definir la estructura del grid
    n_binary = len(binary_columns)
    
    # Crear grid para todos los tipos de gráficos
    gs = gridspec.GridSpec(2, 2)
    
    # 1. Gráficos de barras en la parte superior izquierda
    ax1 = plt.subplot(gs[0, 0])
    
    # Si hay muchas variables binarias, mostrar solo las primeras 5
    display_cols = binary_columns[:min(5, n_binary)]
    
    # Crear un DataFrame para los conteos
    counts_df = pd.DataFrame()
    
    for col in display_cols:
        counts = df[col].value_counts().reset_index()
        counts.columns = ['Value', 'Count']
        counts['Variable'] = col
        counts_df = pd.concat([counts_df, counts])
    
    # Graficar barras agrupadas
    sns.barplot(x='Variable', y='Count', hue='Value', data=counts_df, ax=ax1)
    ax1.set_title('Distribuciones de variables binarias')
    ax1.set_ylabel('Frecuencia')
    
    # 2. Matriz de asociación en la parte superior derecha (usando método apropiado)
    ax2 = plt.subplot(gs[0, 1])
    
    # Calcular la matriz de asociación para variables binarias
    binary_df = df[binary_columns]
    
    # Construir la matriz de asociación manualmente
    n = len(binary_columns)
    association_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # La diagonal es siempre 1
                association_matrix[i, j] = 1.0
            else:
                # Crear tabla de contingencia
                contingency_table = pd.crosstab(df[binary_columns[i]], df[binary_columns[j]])
                
                if association_method == 'phi':
                    # Coeficiente Phi
                    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
                    n_samples = contingency_table.sum().sum()
                    phi = np.sqrt(chi2 / n_samples)
                    association_matrix[i, j] = phi
                
                elif association_method == 'cramers_v':
                    # V de Cramer
                    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
                    n_samples = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n_samples * min_dim))
                    association_matrix[i, j] = cramers_v
                
                elif association_method == 'pearson':
                    # Correlación de Pearson (menos adecuada para binarias)
                    correlation = df[binary_columns[i]].corr(df[binary_columns[j]], method='pearson')
                    association_matrix[i, j] = correlation
    
    # Crear DataFrame para la visualización
    association_df = pd.DataFrame(
        association_matrix, 
        index=binary_columns, 
        columns=binary_columns
    )
    
    method_names = {
        'phi': 'Coeficiente Phi',
        'cramers_v': 'V de Cramer',
        'pearson': 'Correlación de Pearson'
    }
    
    # Visualizar matriz de asociación
    sns.heatmap(association_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    ax2.set_title(f'Matriz de asociación ({method_names.get(association_method, association_method)})')
    
    # 3. Gráficos circulares en la parte inferior izquierda
    ax3 = plt.subplot(gs[1, 0])
    
    # Crear un gráfico circular para una variable binaria representativa
    # o se puede seleccionar una variable específica de interés
    if n_binary > 0:
        target_col = binary_columns[0]  # Usamos la primera variable como ejemplo
        
        value_counts = df[target_col].value_counts()
        labels = [f"{val} ({count})" for val, count in zip(value_counts.index, value_counts.values)]
        ax3.pie(value_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax3.axis('equal')  # Para que el círculo sea un círculo
        ax3.set_title(f'Distribución de {target_col}')
    
    # 4. Tabla de contingencia para 2 variables binarias en la parte inferior derecha
    ax4 = plt.subplot(gs[1, 1])
    
    if n_binary >= 2:
        # Seleccionar las dos primeras variables binarias
        col1, col2 = binary_columns[:2]
        
        # Crear tabla de contingencia
        contingency = pd.crosstab(df[col1], df[col2], normalize='all') * 100
        
        # Visualizar como heatmap
        sns.heatmap(contingency, annot=True, fmt='.1f', cmap='Blues', ax=ax4)
        ax4.set_title(f'Tabla de contingencia: {col1} vs {col2}')
    else:
        ax4.text(0.5, 0.5, "Se necesitan al menos 2 variables binarias\npara una tabla de contingencia",
                horizontalalignment='center', verticalalignment='center')
        ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

# Función modificada para calcular medidas de asociación apropiadas para variables binarias
def plot_binary_association_matrix(df, method='phi', figsize=(10, 8)):
    """
    Crea una matriz de asociación para variables binarias usando medidas apropiadas
    
    Parameters:
    df : pandas DataFrame
    method : str, método de asociación ('phi', 'cramers_v', 'chi_square', 'pearson')
    figsize : tuple, tamaño de la figura
    """
    binary_columns = df.columns[df.nunique() == 2].tolist()
    
    if not binary_columns:
        print("No se encontraron variables binarias en el dataset.")
        return
    
    n = len(binary_columns)
    association_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # La diagonal es siempre 1
                association_matrix[i, j] = 1.0
            else:
                # Crear tabla de contingencia
                contingency_table = pd.crosstab(df[binary_columns[i]], df[binary_columns[j]])
                
                if method == 'phi':
                    # Coeficiente Phi para variables binarias (2x2)
                    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
                    n_samples = contingency_table.sum().sum()
                    phi = np.sqrt(chi2 / n_samples)
                    association_matrix[i, j] = phi
                
                elif method == 'cramers_v':
                    # V de Cramer (generalización de Phi)
                    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
                    n_samples = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n_samples * min_dim))
                    association_matrix[i, j] = cramers_v
                
                elif method == 'chi_square':
                    # Valor p del test Chi-cuadrado
                    _, p_value, _, _ = chi2_contingency(contingency_table)
                    # Convertir p-value a una medida de asociación (1-p)
                    association_matrix[i, j] = 1 - p_value
                
                elif method == 'pearson':
                    # Correlación de Pearson (menos recomendado para binarias)
                    correlation = df[binary_columns[i]].corr(df[binary_columns[j]], method='pearson')
                    association_matrix[i, j] = correlation
    
    # Crear DataFrame para la visualización
    association_df = pd.DataFrame(
        association_matrix, 
        index=binary_columns, 
        columns=binary_columns
    )
    
    # Visualización
    plt.figure(figsize=figsize)
    sns.heatmap(association_df, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    
    method_names = {
        'phi': 'Coeficiente Phi',
        'cramers_v': 'V de Cramer',
        'chi_square': 'Significancia Chi-cuadrado (1-p)',
        'pearson': 'Correlación de Pearson'
    }
    
    plt.title(f'Matriz de asociación de variables binarias ({method_names.get(method, method)})')
    plt.show()
    
    return association_df
