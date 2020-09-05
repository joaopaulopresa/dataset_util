
import pandas as pd

def load_data_set(path,features):
    df = pd.read_excel(path, sheet_name='Planilha1')
    df = clean_data_set(df,features)
    return df

def clean_data_set(df,features):
    all_features = ['Reclamação','Elogio','Sugestao','L.A.I.','Denúncia']
    filter_features = [f for f in all_features if f not in features]
    # remove classes that will not be used
    filtro = df['tipo_manifestacao'] != 'Outros'
    df = df[filtro]
    filtro = df['tipo_manifestacao'] != 'Solicitação'
    df = df[filtro]
    filtro = df['tipo_manifestacao'] != 'Informação Geral'
    df = df[filtro]
    filtro = df['tipo_manifestacao'] != 'Comunicação'
    df = df[filtro]
    filtro = df['classificacao'] != 'Teste'
    df = df[filtro]
    filtro = df['sub_classificacao'] != 'Teste'
    df = df[filtro]
    filtro = df['sub_classificacao'] != 'Repetida/teste'
    df = df[filtro]
    for f in filter_features:
        filtro = df['tipo_manifestacao'] != f
        df = df[filtro]
    # remove null values
    df = df.dropna(subset=['desc_fato'])
    df = df.dropna(subset=['tipo_manifestacao'])
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)],
            axis=1,
            inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove data duplicates
    df = df.drop_duplicates('desc_fato') 
    df.reset_index(drop=True, inplace=True)

    # remove texts that contain only numbers
    df = df.loc[df.desc_fato.apply(lambda x: not isinstance(x, (float, int)))]

    # remove blanks
    df = df.loc[df.desc_fato.apply(lambda x: not x.isspace())]
    df = df[['tipo_manifestacao','desc_fato']]
    return df
    