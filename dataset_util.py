
import pandas as pd

import re
import nltk
from bs4 import BeautifulSoup
from string import punctuation
import unicodedata
import string 
from gensim import utils
import gensim.parsing.preprocessing as gsp

nltk.download('punkt')

sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
remove_arr = ['(ais)','(â)','(eja)','(ª)','(os)','(es)','(o)','( a)','(S)','(m)','(ã)','(eis)','(ões)','(is)','(íram)','(as)','(a)','(ão)','(s)']
# remove accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii',
                                                      'ignore').decode(
                                                          'utf-8', 'ignore')
    return text

def remove_oc(s):
    for e in remove_arr:
        s = s.replace(e, '')
    return s
filters = [
           gsp.strip_tags, 
           #gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           #gsp.remove_stopwords, 
           #gsp.strip_short, 
           #gsp.stem_text
          ]
filters_full = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           remove_accented_chars,
           #gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]

def clean_text_gensim(s,full):
    #s = s.lower()
    for e in remove_arr:
        s = s.replace(e, '')
    s = utils.to_unicode(s)
    filtro = filters
    if full:
        filtro = filters_full
    for f in filtro:
        s = f(s)
    return s
def pre_processing_gensim(df,column,full=False):
    df[column] = df.desc_fato.apply(lambda x: clean_text_gensim(x,full))
    return df
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


def pre_processing_nilc(df,column):
    df[column] = df.desc_fato.apply(clean_text)
    return df

def clean_text(text):
    

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
    for e in remove_arr:
        text = text.replace(e, '')
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
    
def ouvidoria_preprocessing_desc_fato(df,column,
                   html_stripping=True,
                   accented_char_removal=True,
                   remove_extra_space=True,
                   remove_punctuation=True,
                   remove_num=True,
                   remove_emails=True,
                   remove_links=True,
                   lowercase=True):
    df['desc_fato'] = df.desc_fato.apply(remove_oc)
    if html_stripping:
            df[column] = df.desc_fato.apply(strip_html_tags)

    # remove accented characters
    if accented_char_removal:
        df[column] = df[column].apply(remove_accented_chars)
    
    # remove emails 
    if remove_emails:
        df[column] = df[column].apply(remove_email)
    
    # remove links
    if remove_links:
        df[column] = df[column].apply(remove_website_links)

    if remove_num:
        df[column] = df[column].apply(remove_numbers)

    #add space between punctuation
    df[column] = df[column].apply(add_space)

    # remove punctuation
    if remove_punctuation:
        df[column] = df[column].apply(strip_punctuation)
    
     # remove extra whitespace
    if remove_extra_space:
        df[column] = df[column].apply(remove_extra_whitespace)
    
    if lowercase:
       df[column] = df[column].apply(lambda x:  x.lower())

    return df
# strip HTML
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text




# remove extra whitespace
def remove_extra_whitespace(text):
    return re.sub(' +', ' ', text)

# remove punctuation
def strip_punctuation(text):
    return ''.join(c for c in text if c not in punctuation)

#remove Numbers
def remove_numbers(text):
    return ''.join(c for c in text if not c.isdigit())

#remove emails 
def remove_email(text):
    return re.sub(r"\S*@\S*\s?", ' ', text)

def add_space(text):
    return text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))

# remove links
def remove_website_links(text):
    return re.sub(r"http\S+", ' ', text)