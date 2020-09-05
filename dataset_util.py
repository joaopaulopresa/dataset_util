
import pandas as pd

import re
import nltk

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
    df['desc_fato'] = df['desc_fato'].astype(str)
    return df


def pre_processing_nilc(df):
    df['desc_fato'] = df.desc_fato.apply(clean_text)
    return df

def clean_text(text):
    nltk.download('punkt')

    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')

    # Punctuation list
    punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')

    # ##### #
    # Regex #
    # ##### #
    re_remove_brackets = re.compile(r'\{.*\}')
    re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
    re_transform_numbers = re.compile(r'\d', re.UNICODE)
    re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
    re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
    # Different quotes are used.
    re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
    re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
    re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
    re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
    re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
    re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
    re_tree_dots = re.compile(u'…', re.UNICODE)
    # Differents punctuation patterns are used.
    re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                        (punctuations, punctuations), re.UNICODE)
    re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                            (punctuations, punctuations), re.UNICODE)
    re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
    re_changehyphen = re.compile(u'–')
    re_doublequotes_1 = re.compile(r'(\"\")')
    re_doublequotes_2 = re.compile(r'(\'\')')
    re_trim = re.compile(r' +', re.UNICODE)
    """Apply all regex above to a given string."""
    text = text.lower()
    text = text.replace('\xa0', ' ')
    text = re_tree_dots.sub('...', text)
    text = re.sub('\.\.\.', '', text)
    text = re_remove_brackets.sub('', text)
    text = re_changehyphen.sub('-', text)
    text = re_remove_html.sub(' ', text)
    text = re_transform_numbers.sub('0', text)
    text = re_transform_url.sub('URL', text)
    text = re_transform_emails.sub('EMAIL', text)
    text = re_quotes_1.sub(r'\1"', text)
    text = re_quotes_2.sub(r'"\1', text)
    text = re_quotes_3.sub('"', text)
    text = re.sub('"', '', text)
    text = re_dots.sub('.', text)
    text = re_punctuation.sub(r'\1', text)
    text = re_hiphen.sub(' - ', text)
    text = re_punkts.sub(r'\1 \2 \3', text)
    text = re_punkts_b.sub(r'\1 \2 \3', text)
    text = re_punkts_c.sub(r'\1 \2', text)
    text = re_doublequotes_1.sub('\"', text)
    text = re_doublequotes_2.sub('\'', text)
    text = re_trim.sub(' ', text)
    return text.strip()
    