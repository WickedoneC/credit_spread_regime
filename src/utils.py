def prettyTable(df, col):
    a = df[col].value_counts(dropna = False)
    b = df[col].value_counts(normalize = True, dropna = False)
    c = pd.concat([a, b], axis = 1)
    c.columns = [col + '_count', col + '_ratio']
    return c