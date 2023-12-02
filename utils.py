import pandas as pd

# Compute max length for each data split
def max_length_split(df:pd.DataFrame,column1: str, column2: str, split: str ):
    
    conclusion_lengths = df[column1].str.len()
    premise_lengths = df[column2].str.len()

    max_length_c = conclusion_lengths.max() 
    max_length_p = premise_lengths.max()

    print(f"- Maximum length of the pair {column1} | {column2} in the {split} set is respectively\
 ({max_length_c},{max_length_p}) characters")
    # return max_idx



## Count outlier
def outliers_per_unique_conclusion(counts_df: pd.DataFrame,df_args,threshold:int,split: str):
    nb_outliers = len(counts_df[counts_df['Count'] > threshold])
    nb_unique_conclusions = len(df_args['Conclusion'].unique())
    print(f'- Percentage of outliers per unique conclusions in the {split} set:\
 {nb_outliers / nb_unique_conclusions * 100:.2f}%')


########
def count_repeated_conclusions(split,df_args):
    """_summary_

    Args:
        split (_type_): _description_
        df_args (_type_): _description_
    """    
    len_args = len(df_args)
    unique_conclusion_split = len(df_args['Conclusion'].unique())
    print(f'- There ratio of unique conclusion in the {split} set is \
{unique_conclusion_split / len_args*100:.2f}%')


def count_problematic(df_labels: pd.DataFrame, df_args: pd.DataFrame, value: str):
    w_value = df_labels.loc[df_labels[value] == 1]
    w_value_sum = w_value.sum(axis=1, numeric_only=True)
    cnt_probl_value = w_value_sum[w_value_sum == 1].count()

    absolute_count = cnt_probl_value
    relative_percentage =  (cnt_probl_value / len(df_args))
    return absolute_count, relative_percentage

def cnt_probl_per_split_df(split, split_lbl, split_args, values):
    data = {'Split': [], 'Aspect': [], 'Absolute': [], 'Relative': []}

    for value in values:
        absolute_count, relative_percentage = count_problematic(split_lbl, split_args, value)
        data['Split'].append(split)
        data['Aspect'].append(value)
        data['Absolute'].append(absolute_count)
        data['Relative'].append(relative_percentage)

    df = pd.DataFrame(data)
    pivot = df.pivot_table(index=['Split', 'Aspect'], values=['Absolute', 'Relative'], aggfunc='sum')
    pivot['Relative'] = pivot['Relative'].apply(lambda x: f"{x:.2%}") # Round up to two digits and append percentage sign
    return pivot


def get_value_sum(value:str,df_labels:pd.DataFrame):
    row_with_value = df_labels.loc[df_labels[value] == 1] # select those rows
    row_with_value_agg = row_with_value.sum(axis = 1, numeric_only = True)
    
    return row_with_value, row_with_value_agg

def check_collision(value:str, values:list[str],df:pd.DataFrame,df_sum:pd.DataFrame):
    value_check = df[df_sum == 2][values]
    for other_value in values:
        if other_value != value:
            assert (value_check[other_value] == 0).all()
    return value_check 

def to_level_3(df):
    lvl3_labels = pd.DataFrame()
    lvl3_labels['Argument ID'] = df['Argument ID']

    cols_groups = {
        'openness_to_change': ['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism'],
        'self_enhancement': ['Hedonism', 'Achievement', 'Power: dominance', 'Power: resources', 'Face'],
        'conservation': ['Face', 'Security: personal', 'Security: societal', 'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility'],
        'self_transcendence': ['Humility', 'Benevolence: caring', 'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance', 'Universalism: objectivity'],
    }

    for cat, cols in cols_groups.items():
        filtered = df.filter(items=cols)
        lvl3_labels[cat] = filtered.any(axis=1).astype(int)

    return lvl3_labels



def lbl_co_occurrance(df_labels: pd.DataFrame ,label_list:list[str]):
    df_labels_occ = df_labels.copy()
    df_labels_occ.head()
    labels = df_labels[label_list]
    df_labels_occ['label_combination'] = labels.apply(lambda row: tuple(row), axis=1)
    
    # Count occurrences of each label combination for each Premise
    premise_label_counts = df_labels_occ.groupby(['label_combination']).size().reset_index(name='count')

    return premise_label_counts


def cnt_percentage_label_alone(df_count:pd.DataFrame, indexes: list[int],labels:pd.DataFrame) -> float:
    labels_alone = df_count.iloc[indexes].copy()

    cnt_total_single_label = labels_alone['count'].sum()

    percentage_single_label = cnt_total_single_label / len(labels)
    
    labels_alone_no_idx = labels_alone.reset_index(drop=True).copy()
    # Reset index is need to not carry over the indexes from the df_count dataframe
    return labels_alone_no_idx, percentage_single_label

def cnt_pctg_alone_split(df_lbls_alone:pd.DataFrame,df_pctg:pd.DataFrame,split:str):
    df_pctg_split = df_pctg[split]
    count_alone = df_lbls_alone['count']
    division_df = pd.DataFrame()
    division_df['Category'] = df_pctg['Category']
    division_df['Percentage'] = count_alone / df_pctg_split
    division_df['Percentage'] = division_df['Percentage'].apply(lambda x: f'{x:.2%}')

    return division_df