import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_distribution_by_category(df, categorical_col, max_cols=3, show_stats=True):
    """
    Plots the distribution of numerical features within each category of a categorical column,
    with vertical lines for mean and median, and optionally shows stats next to the legend.

    Parameters:
    df (pd.DataFrame): The DataFrame containing numerical and categorical data.
    categorical_col (str): The name of the categorical column to use as hue.
    max_cols (int): Maximum number of columns in the subplot grid.
    show_stats (bool): Whether or not to display the mean and median stats in the legend.

    Returns:
    None
    """
    numerical_cols = df.select_dtypes(include=['number']).columns
    num_vars = len(numerical_cols)

    # Define grid size for subplots
    num_rows = int(np.ceil(num_vars / max_cols))
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(max_cols * 6, num_rows * 5))
    axes = axes.flatten()

    palette = sns.color_palette("tab10", n_colors=df[categorical_col].nunique())  # Unique colors for categories
    for i, col in enumerate(numerical_cols):
        ax = axes[i]

        # KDE plot with hue
        sns.kdeplot(data=df, x=col, hue=categorical_col, fill=True, common_norm=False, alpha=0.3, palette=palette, ax=ax)

        # Store legend handles and labels for later use
        legend_handles = []
        legend_labels = []

        # Compute mean and median for each category and plot
        for j, category in enumerate(df[categorical_col].unique()):
            subset = df[df[categorical_col] == category][col].dropna()
            mean_value = subset.mean()
            median_value = subset.median()

            # Add vertical lines with matching color
            mean_line = ax.axvline(mean_value, color=palette[j], linestyle='--', alpha=0.9, linewidth=2)
            median_line = ax.axvline(median_value, color=palette[j], linestyle='-', alpha=0.9, linewidth=2)

            # Add to legend info only if stats are requested
            if show_stats:
                legend_handles.append(mean_line)
                legend_handles.append(median_line)
                legend_labels.append(f'{category} Mean: {mean_value:.2f}')
                legend_labels.append(f'{category} Median: {median_value:.2f}')
            else:
                # Add only the mean/median lines without stats if show_stats=False
                legend_handles.append(mean_line)
                legend_handles.append(median_line)
                legend_labels.append(f'{category} Mean')
                legend_labels.append(f'{category} Median')

        # Title & labels
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)

        # Add individual legend to each subplot
        ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize=9, title="Category Stats")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'../img/distribution_by_category_{categorical_col.replace("/", "_")}.png')
    plt.show()

# Example usage:
# plot_distribution_by_category(df, 'y', show_stats=True)  # To show stats









def plot_correlation_by_category(df, categorical_col, max_cols=3):
    """
    Plots correlation matrices for numerical features within each category of a categorical column,
    with square sizes proportional to correlation strength.

    Parameters:
    df (pd.DataFrame): The DataFrame containing numerical and categorical data.
    categorical_col (str): The name of the categorical column to group by.
    max_cols (int): Maximum number of columns in the subplot grid.

    Returns:
    None
    """
    current_time = time.strftime("%Y%m%d-%H%M%S")
    categories = df[categorical_col].unique()
    num_categories = len(categories)

    # Define grid size for subplots
    num_rows = int(np.ceil(num_categories / max_cols))
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(max_cols * 6, num_rows * 5))
    axes = axes.flatten()  # Flatten to handle indexing easily

    for i, category in enumerate(categories):
        subset = df[df[categorical_col] == category].select_dtypes(include=['number'])

        # Skip categories with less than 2 numerical columns
        if subset.shape[1] < 2:
            print(f"Skipping {category}: Not enough numerical features for correlation analysis.")
            continue

        corr = subset.corr()

        # Define marker size dynamically based on absolute correlation values
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask upper triangle for cleaner visualization
        square_size = np.abs(corr.values)  # Use absolute values to scale sizes

        ax = axes[i]
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.75, "label": "Correlation"},
            mask=mask, ax=ax, square=True
        )

        # Improve title readability
        num_samples = len(subset)
        ax.set_title(f'{categorical_col} = {category} (n={num_samples})', fontsize=12, fontweight='bold')

        # Rotate x/y labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)


    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'../img/correlation_by_category_{categorical_col.replace("/", "_")}_{current_time}.png')

    plt.show()
    # SAve Figure, replcae all / characters with _ to avoid errors on categorical_col

# Example usage:
# plot_correlation_by_category(df, 'y')


