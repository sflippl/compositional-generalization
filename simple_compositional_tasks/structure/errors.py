def check_column(df, column, arg_name):
    if column not in df.columns:
        raise ValueError(f'{arg_name} has no column {column}.')