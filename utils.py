import numpy as np
import pandas as pd

def get_data_summary(data,target_col=None):
    target_data = data[target_col] if isinstance(target_col,str) else target_col
    get_col_summary = lambda col,data_type: {
        'ColumnName': col.name, 'DataType': data_type, 'NumberOfMissingValues': col.isnull().sum(),
        'CorrelationWithTarget': col.corr(target_data) if data_type not in ['category','datetime64[ns]','object'] and target_data is not None else np.nan,
        'Mean': col.mean() if data_type not in ['category','binary','object'] else np.nan, 
        'Median': col.median() if data_type not in ['category','object','datetime64[ns]']  else np.nan,
        'Mode': col.mode().values[0],
        'MinValue': col.min() if data_type not in ['category','binary','object'] else np.nan, 
        'MaxValue': col.max() if data_type not in ['category','binary','object'] else np.nan,
        'NumberOfUniqueValues': col.nunique(),
        'UniqueValues': np.sort(col.unique()) if data_type in ['category','binary','object'] else np.nan,
        'FracUniqueVals': list((col.value_counts()/len(col)).round(3) ) if data_type in ['category','binary','object'] else np.nan,
        
    }
    summary_rows = [get_col_summary(data[col],'binary' if data[col].isin([0, 1, np.nan]).all() else str(data[col].dtype)) for col in data.columns]
    info_df = pd.DataFrame(summary_rows, columns=summary_rows[0].keys()).set_index('ColumnName')
    info_df = info_df.sort_values(['CorrelationWithTarget','DataType','ColumnName'], ascending=[False,True,True], na_position='last')
    info_df = info_df.drop("CorrelationWithTarget",axis=1) if target_data is None else info_df
    if target_col in info_df.index:
      info_df = info_df.loc[[ind for ind in info_df.index if ind!=target_col]].append(info_df.loc[target_col])
      info_df.index = [ind for ind in info_df.index if ind!=target_col] + [target_col+"*"] 
    return info_df
    