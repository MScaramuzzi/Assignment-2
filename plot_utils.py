import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


## Boxplot ##########
def boxplot_conclusion_repetition(df:pd.DataFrame):

    conclusion_counts = df['Conclusion'].value_counts() # Count the occurrences of each conclusion
    # Create a DataFrame from the counts
    counts_df = pd.DataFrame({'Conclusion': conclusion_counts.index, 'Count': conclusion_counts.values})

    # Create a boxplot using seaborn with pastel colors
    plt.figure(figsize=(20, 6))
    sns.set(style="whitegrid")

    # Define pastel color palette
    pastel_colors = sns.color_palette("pastel")

    # Create ax object
    ax = sns.boxplot(x='Count', data=counts_df, palette=pastel_colors)

    # Calculate quartiles
    q1 = counts_df['Count'].quantile(0.25)
    median = counts_df['Count'].median()
    q3 = counts_df['Count'].quantile(0.75)
    max_repetitions =  counts_df['Count'].max() 

    # Highlight quartiles with vertical lines
    ax.axvline(x=q1, color='g', linestyle='--', label=f'Q1: {q1:.2f}')
    ax.axvline(x=median, color='b', linestyle='--', label=f'Median: {median:.2f}')
    ax.axvline(x=q3, color='r', linestyle='--', label=f'Q3: {q3:.2f}')

    # Set labels and title
    ax.set(xlabel='Count', title='Boxplot of Conclusion Counts')
    plt.xticks(np.arange(0,max_repetitions,q3))
    plt.legend()
    plt.show()
    return counts_df


def plot_freq_lbl_w_percentage(label_df_train:pd.DataFrame,
 label_df_val: pd.DataFrame, lbl_cols:list[str]) -> dict:
    # List of splits and corresponding colors
    splits = [label_df_train, label_df_val]
    colors = ['red', 'blue']

    # Plot histogram for each label in a single plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    dict_count = {label: {} for label in lbl_cols}
    
    for j, split in enumerate(splits):
        for i, label in enumerate(lbl_cols):
            
            class_count = split[label].sum()
            dict_count[label][f'{["Train", "Validation"][j]}'] = class_count
            percentage = (class_count / len(split)) * 100  # Calculate percentage

            plt.bar(i + 0.2 * j, class_count, width=0.2, color=colors[j], alpha=0.7,
                    label=f'{["Train", "Validation"][j]}' if i == 0 else "")
            
            # Add percentage text inside the bar
            plt.text(i + 0.2 * j, class_count, f'{percentage:.1f}%', ha='center', va='bottom',
                     color='black', fontsize=13, fontweight='bold')

    plt.title('Frequency of labels for each data split')
    plt.xlabel('Labels')
    plt.ylabel('Frequency of labels')
    plt.xticks(np.arange(len(lbl_cols)) + 0.2, labels=lbl_cols)
    plt.legend()

    # Remove vertical grid lines
    plt.gca().xaxis.grid(False)
    plt.show()
    
    return dict_count



def plot_f1_w_distribution(label_df_train: pd.DataFrame, f1s: dict, macro_f1s: dict, lbl_cols: list[str]) -> None:
    plt.figure(figsize=(14, 6), dpi=300)
    markers = ['o','X','d','*','p']
    sc_colors = ['limegreen','magenta', 'deepskyblue', 'firebrick', 'orange']

    for i, label in enumerate(lbl_cols):
        class_count = label_df_train[label].sum() / len(label_df_train)

        plt.bar(i, class_count, width=0.2, color='red', alpha=0.4, zorder=2)

        for j, (model_name, scores) in enumerate(f1s.items()):
            f1 = scores[i]
            plt.scatter(i, f1, color=sc_colors[j], marker=markers[j],
                        label=f'{model_name}' if i==0 else '', zorder=3)

    # plot line connecting per-class f1s for each model
    for j, scores in enumerate(f1s.values()):   
        plt.plot(range(len(lbl_cols)), scores, color=sc_colors[j], linestyle='-', linewidth=1,
                     zorder=1)

    # Plot macro F1-scores
    for i, avg_f1 in enumerate(macro_f1s.values()):
        plt.scatter(len(lbl_cols), avg_f1, color=sc_colors[i], marker=markers[i])

    plt.ylabel('F1-score')
    plt.xticks(np.arange(len(lbl_cols)+1), labels=lbl_cols + ['Mean'])
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc='upper left')
    plt.grid(linestyle='dashed')
    plt.ylim([0, 1])
    plt.show()



def plot_precision_recall_comparison(precisions: dict, recalls: dict) -> None:
    plt.figure(figsize=(10, 6), dpi=300)
    markers = ['o','X','d','*','p']
    colors = ['limegreen','magenta', 'deepskyblue', 'firebrick', 'orange']

    for i, model_name in enumerate(precisions.keys()):
        precision = precisions[model_name]
        recall = recalls[model_name]

        plt.scatter(precision, recall,
                    marker=markers[i], color=colors[i], label=f'{model_name}')

    plt.title('Macro Average Precision-Recall Comparison')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.grid(linestyle='dashed')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
