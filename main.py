from typing import List, Set, Dict, Tuple, Optional, Any
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import spacy
from spacy.tokens import Token
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # can generate clouds as images.
import pickle
import numpy as np
import pandas as pd
import os
from pprint import pprint
from pprint import pformat
import re
import logging
import sys

"""
Errors to look out for:
- some problem in general wordclouds at saving to file - switching off for the moment
- number of texts smaller than number of files - if that happens one text file must be empty, check by file size
"""

# Program switches
corpus = 'sales'
process_texts = False  # if False, the corpus will be loaded from processed gensim file
make_wordclouds = False  # these are general, for the whole corpus, False saves time.
corpus_analysis = False  # generates counts for wordlists
lda_modelling = True  # runs a series of LDA models
filter_bad_tokens = True  # removes hand picked tokens from dicitonary and re-runs text to BOW
lda_wordclouds = False  # produces a wordcloud for each LDA topic - time consuming
forecast_probabilities = False  # calculates document probabilities on the same corpus, assuming modelling was done.
forecast_probabilities_sd = False  # calculates document probabilities for management report corpus

# Advanced options
make_stoplist_regex = False  # if False, script will  attempt to load the stoplist
drop_stoplist_regex = True  # removes the stoplist regext to reduce processing time
filter_extremes = False  # be careful with small corpora here, check parameters
filter_stoplist = True  # filters gensim dictionary to remove stoplist instead of regexing

if process_texts:
    corpus_load = False
else:
    corpus_load = True

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', filename=str("log_" + corpus + ".txt"))
# consider logging all console output using this: https://stackoverflow.com/questions/70542526/how-to-save-python-script-console-output-to-a-file


# Global variables
my_stoplist: str = ''

if corpus == 'forecasts':
    corpus_name = 'sd_forecasts'
    source_data = './data/sd_forecasts'
    results_folder = './results/sd_forecasts'
    text_encoding = "utf-8"
elif corpus == 'operations':
    corpus_name = 'sd_operations'
    source_data = './data/sd_operations'
    results_folder = './results/sd_operations'
    text_encoding = "utf-8"
elif corpus == 'sales':
    corpus_name = 'sd_sales'
    source_data = './data/sd_sales'
    results_folder = './results/sd_sales'
    text_encoding = "utf-8"
elif corpus == 'sobolewski':
    corpus_name = 'sobolewski'
    source_data = './data/sobolewski'
    results_folder = './results/sobolewski'
    text_encoding = "utf-8"
elif corpus == 'sd_iaaer':
    corpus_name = 'sd_iaaer'
    source_data = './data/sd_iaaer'
    results_folder = './results/sd_iaaer'
    text_encoding = "utf-8"
elif corpus == 'la_iaaer':
    corpus_name = 'la_iaaer'
    source_data = './data/la_iaaer'
    results_folder = './results/la_iaaer'
    text_encoding = "utf-8"
else:
    print('Wrong corpus name selected in program switches, exiting...')
    sys.exit(1)

# ## STOP LIST ## #


def make_exclusion_regex(filename, encoding):
    '''
    Function generates a regex to use with is_excluded from a stop-list file
    :param filename: file with stop-list
    :param encoding: encoding of the stop-list file
    :return: string of regex groups marked for match from beginning to end.
    '''
    input = []
    with open(filename, "rt", encoding=encoding) as f:
        for line in f:
            input.append(str(line.strip()).lower())
    output = str("(\Aproc\Z)")  # "proc" is percent, i.e. "proc."
    for w in input:
        output = "|".join((output, str("(\A" + w + "\Z)")))
    print('Exclusions regex prepared, string length: ' + str(len(output)))
    return output


if make_stoplist_regex:  # this regex will pass into the NLP pipeline, removing tokens at a time
    my_stoplist = make_exclusion_regex("stop_lista_kombo_utf8.txt", "utf-8")

    with open('exclusion_regex.txt', 'wt', encoding='utf-8') as f:
        f.write(my_stoplist)
    with open('exclusion_regex.pk', 'wb') as f:
        pickle.dump(my_stoplist, f)
elif drop_stoplist_regex:  # this removes stoplist from get_excluded to speed up text processing
    my_stoplist = ""
else:
    try:
        with open('exclusion_regex.pk', 'rb') as f:
            my_stoplist = pickle.load(f)
    except:
        print('Error loading stoplist file, create it first by changing the make_stoplist switch to True')
        sys.exit(1)


def make_stoplist_iterable(filename, encoding):
    input = []
    with open(filename, "rt", encoding=encoding) as f:
        for line in f:
            input.append(str(line.strip()).lower())
    output = input
    return output


# ### LOAD SPACY NLP ### #

# first run download the model via terminal: python -m spacy download pl_core_news_sm
# can change to more accurate but slower model "pl_core_news_lg"
nlp = spacy.load("pl_core_news_sm")


# ## UTILITY FUNCTIONS ## #

def get_gensim_ids(list_of_lemmas, gensim_dictionary):
    """
    Generates a set of ids from gensim.dictionary.token2id.
    :param list_of_lemmas: any list of strings that may appear in the dictionary
    :param gensim_dictionary: a gensim.dictionary object
    :return: list of gensim ids
    """
    g_dict_keys = [k for k in gensim_dictionary.token2id.keys()]
    g_dict_keys_count = {k: g_dict_keys.count(k) for k in g_dict_keys}
    ids = set()
    for lemma in list_of_lemmas:
        try:
            count = g_dict_keys_count[lemma]
        except:
            continue
        if count == 1:
            ids.add(gensim_dictionary.token2id.get(g_dict_keys[g_dict_keys.index(lemma)]))
        else:
            templist = g_dict_keys
            baseindex = 0
            for x in range(0, count):
                index = templist.index(lemma)
                baseindex = baseindex + index  # index used for accessing ids in gensim dictionary
                ids.add(gensim_dictionary.token2id.get(g_dict_keys[baseindex]))
                templist = templist[index + 1:len(templist) - 1]  # slicing list to search the rest
                baseindex += 1  # this takes care of changing index when list is sliced
    return ids


def corpus_scoring(word_list, gensim_dict):
    """
    Produces a dataframe with frequencies per lemma from word_list based on a corpus.
    :param word_list: a list of strings, lemmas
    :param gensim_dict: a dictionary produced by gensim from a corpus
    :return: dataframe
    """
    freq = []
    for lemma in word_list:
        score: int = 0
        try:
            score = gensim_dict.cfs[gensim_dict.token2id[lemma]]
        except:
            score = 0  # if lemma not in dictionary
        freq.append(score)
    df_out = pd.DataFrame({'Lemma': word_list, 'Frequency': freq})
    df_out['freq_ratio'] = df_out['Frequency'] / gensim_dict.num_nnz
    return df_out


def descriptiveStatistics(title, freq_array, **kwargs):  # CHEEEECK THISSSSS
    """
    Function compares sentiment indexes for two lists using simple statistics and correlations.
    :param title: String to be included in output.
    :param new_quality_list: Array-like object with single vector in interval scale (eg. floats).
    :param old_quality_list: Array-like object with single vector in interval scale (eg. floats).
    :param kwargs: Optional parameters to trigger switches.
    :return: person correlation coefficient and anova results
    """
    print("***Descriptive statistics for " + title + ".***")
    temp_list = []
    for x in range(len(freq_array)):
        if np.isnan(freq_array[x]):  # remove NaN rows.
            continue
        else:
            temp_list.append(freq_array[x])
    df = pd.DataFrame({"wordlist": temp_list})
    print(df.describe())
    return 0


def word_list_parser(raw_text):
    """
    Takes in a list as raw texts and creates a List object with no duplicates.
    Improve this in the future to increase flexiblity (e.g. various delimiters).
    :param raw_text: any list
    :return: List object without duplicates
    """
    output: List[str] = []
    list_of_lemmas = raw_text.split(",")  # default delimiter
    if len(list_of_lemmas) == 0:
        list_of_lemmas = raw_text.split(";")  # try if comma didn't work
    if len(list_of_lemmas) == 0:
        list_of_lemmas = raw_text.split(" ")  # try if nothing worked
    output = list(set([(str(l).lower()).strip() for l in list_of_lemmas]))
    return output


def read_texts(files_list, encoding):
    """
    Basic processing from text files into strings.
    :param files_list: list of files to process - i.e. paths that can be passed to os
    :param encoding: text file encoding
    :return: list of strings
    """
    output = []
    counter = 0
    for fl in files_list:
        with open(fl, "rt", encoding=encoding) as f:
            text = str()
            for line in f:
                text = " ".join((text, str(line.strip()).lower()))
        output.append(text)
        counter = counter + 1
    print("Done reading a total of %d text files." % counter)
    return output


def get_is_excluded(token):
    """
    Modifies the Token class to exclude arbitrary tokens that you want to remove.
    WARNING: the function relies on global variable my_stoplist (cannot be passed as argument)
    WARNING: if my_stoplist is empty, it will be skipped.
    :param token: Spacy.Token
    :return: Spacy.Token.is_excluded property
    """
    is_excluded = False
    if token.is_stop or token.is_punct or token.ent_type_ or token.lemma_ == '-PRON-':
        is_excluded = True
    if len(token.text) < 4 or token.is_alpha is False:
        is_excluded = True
    if len(my_stoplist)>0:
        if re.search(my_stoplist, token.lemma_):
            # WARNING: the function relies on global variable my_stoplist (cannot be passed as argument)
            is_excluded = True
    return is_excluded


# This line is necessary to get the get_is_excluded() function working with the Token class:
Token.set_extension('is_excluded', getter=get_is_excluded)


def corpuser(texts_list):
    """
    Pre-process texts using spaCy pipeline defined earlier.
    :param texts_list: a list of strings
    :return: list of spaCy docs.
    """
    output = []
    for t_raw in texts_list:
        t_out = str(t_raw).strip().lower()
        t_out = ' '.join([word if word.isalpha() else word[:-1] if word[:-1].isalpha() else word[
                                                                                            1:] if word[
                                                                                                   1:].isalpha() else ""
                          for word in t_out.split()])
        output.append(nlp(t_out))
    return output


def gensimmer(list_of_docs):
    """
    Process spaCy docs into lists of strings, excluding unwanted tokens and short docs.
    :param list_of_docs: a list of spacy docs
    :return: list of string texts.
    """
    output = []
    for doc in list_of_docs:
        new_text = []
        for token in doc:
            if not token._.is_excluded:
                new_text.append(token.lemma_)
            else:
                pass
        if len(new_text) > 10:
            output.append(new_text)
        else:
            continue
    return output


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """
    Create word cloud in grey scale at 300 dpi for publication
    """
    return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 95)


def black_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return ("hsl(0,100%, 1%)")


def freq_from_gensim(gensim_dictionary):
    dict_out = {}
    for word in gensim_dictionary.token2id.keys():
        freq = gensim_dictionary.cfs[gensim_dictionary.token2id[word]]
        dict_out.update({word: freq})
    return dict_out


def tfidf_token2id(tfidf_corpus, dictionary):
    '''
    This function takes in the generator expression from TfidfModel(corpus)[corpus] and outputs a dictionary of word-tfidf score
    '''
    id2tfidf = {}
    for line in tfidf_corpus:
        for ID, freq in line:
            pair = {dictionary.get(ID): freq}
            id2tfidf.update(pair)
    return id2tfidf


def frequency_dictionary(gensim_dictionary, word_list):
    '''
    Function returns a dictionary of corpus frequencies per each word in word_list.
    :param gensim_dictionary: gensim.corpora.dictionary.Dictionary object
    :param word_list: a list of tokens that should appear in dictionary
    :return: a dictionary of token-frequency pairs
    '''
    output = dict()
    for w in word_list:
        try:
            output[w] = gensim_dictionary.cfs[gensim_dictionary.token2id[w]]
        except:
            output[w] = 0
    return output


def wordcloud_from_dict(freq_dict, title_string):
    wc = WordCloud(background_color='white', width=3000, height=2000, color_func=black_color_func,
                   stopwords=set({'bogdanka', 'enea'})).generate_from_frequencies(
        frequencies=freq_dict)
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Wordcloud for " + title_string)
    # plt.show()  # pops up the graph while running script
    plt.savefig(
        os.path.join('./results/' + corpus_name + '_' + title_string + '.png'),
        format='png',
        dpi=300)
    plt.close()
    return 0


def wordcloud_from_list(wordlist, gensim_dict, title_string):
    '''
    Generate wordcloud from a list of tokens and save to disk
    :param wordlist: any list of tokens
    :param gensim_dict: gensim dictionary object
    :param title_string: title for plot and file name
    :return: DataFrame of frequencies per word
    '''
    dict_wordlist = frequency_dictionary(gensim_dict, wordlist)
    df_wordlist = pd.DataFrame.from_dict(dict_wordlist, orient='index', columns=['frequency']). \
        sort_values(by='frequency', axis=0, ascending=False)
    df_wordlist['freq_ratio'] = df_wordlist['frequency'] / gensim_dict.num_nnz
    df_wordlist.to_csv('./results/' + corpus_name + '_' + title_string + '.csv')
    if df_wordlist['frequency'].sum() == 0:
        print('Error in wordcloud_from_list for ' + str(title_string).replace(" ", "_") + '. Zero sum of frequencies. Returning empty df.')
        df_wordlist['freq_ratio'] = np.NaN
        return df_wordlist
    else:
        wordcloud_from_dict(dict_wordlist, corpus_name + '_' + title_string)
    return df_wordlist


# ## CORE SCRIPT ## #

print("*** Running corpus analysis for " + corpus_name + " ***")

word_lists = {}

word_lists['HenryPos'] = word_list_parser(
    "prawie, osiągnięcie, uzyskanie, dojście, osiągnąć, uzyskać, wygrać, sukces, rezultat, powodzenie, przyrost, "
    "wzrost, progresja, dorosły, korzyść, zysk, wiodący, istotny, ważny, znaczący, wzlot, sukces, powodzenie, "
    "pokonać, zwyciężyć, przezwyciężyć, przezwyciężyć, zwyciężyć, pokonać, deklasować, pobić, przewyższyć, pobicie, "
    "najlepszy, jedyny, wyjątkowy, najkorzystniejszy, zwycięzki, lepiej, dobrze, prawidłowo, skutecznie, "
    "przejrzyście, poprawić, ulepszyć, skorygować, zoptymalizować, optymalizować, duży, większy, niemały, istotny, "
    "znaczący, wydatny, kapitał, pewny, ewidentny, sprawodzony, wiarygodny, namacalny, pewien, pewny, miarodajny, "
    "pewność, wybitny, wydatny, niespotykany, zachęcający, mobilizujący, motywujący, zachęta, mobilizacja, motywacja, "
    "przekroczyć, przewyższyć, przerosnąć, przewyższyć, doskonale, idealnie, wspaniale, znakomicie, wyśmienicie, "
    "ekspansja, ekspansywny, rozwój, progres, rozszerzać, poszerzać, zwiększać, rozwijać, rozbudowywać, dobry, "
    "korzystny, dobro, szęście, korzyść, dobry, wielki, ciężki, rosnąć, wzrastać, zwiększać, powiększać, wysoki, "
    "duży, wysoko, wysoce, intensywnie, wielce, ulepszony, lepszy, lepiej, ulepszenie, udoskonalenie, poprawa, "
    "usprawnienie, ulepszać, poleszać, zmieniać, zwiększenie, powiększenie, zwiększyć, podnieść, zwielokrotnić, "
    "zmaksymalizować, zwiększenie, powiększenie, duży, silny, wysoki, doprowadzić, przeprowadzić, lider, mistrz, "
    "leader, czołowy, eminentny, znamienity, utytułowany, genialny, pozostawić, cieszyć, większość, dominować, "
    "możliwość, przewyższyć, przerosnąć, cieszyć, zadowolony, rad, szczęśliwy, usatysfakcjonowany, dumny, pozytywny, "
    "skuteczność, zręczność, potężny, mocarny, silny, postęp, osiągnięcie, rozwijać, rozrastać, osiągać, promować, "
    "wypromować, doprowadzić, odczytać, rekord, rekordowy, szczyt, szczytowy, dostarczyć, nagroda, laur, wznosić, "
    "wzbijać, wznieść, umiejętność, wiedza, kwalifikacje, zdolność, solidny, gruntowny, rzetelny, solidność, "
    "wytrzymałość, trwałość, mocny, wzmocnić, umocnić, podnieść, wzmocnienie, umocnienie, wzmożenie, osiągnąć, "
    "wygrać, wypracować, uzyskać, udany, zwycięzki, celny, podnosić, zwiększać, podwyższać, rosnąć, wzrastać, "
    "rozwijać, poprawić, polepszyć, podbić, odbić, dobrze, atrakcyjnie, korzystnie, obiecująco, dokonany "
)

word_lists['HenryNeg'] = word_list_parser(
    "złamany, przybity, spadkowy, spadek, zmniejszenie, redukcja, zły, niedobry, niekorzystny, niedochodowy, "
    "niskodochodowy, upadłość, bankructwo, krach, niewypłacalność, plajta, zakwestionowanie, podważenie, "
    "zaprzeczenie, obalenie, zdyskredytowanie, oprotestowanie, wyzwać, wyzwanie, wymagający, zredukowany, mały, "
    "tłumić, wytłumiać, spaść, zjechać, osłabnąć, dekrementacja, zmniejszać, zmniejszenie, chwiejnie, chwiejny, "
    "niestabilny, niepewny, niestabilnie, niepewnie, depresja, kryzys, krach, recesja, zapaść, dekoniunktura, spadek, "
    "zniżka, bessa, deflacja, pogorszyć, ochłodzić, zepsuć, popsuć, zubożyć, trudny, niełatwy, nielekki, ciężki, "
    "uciążliwy, kłopotliwy, zawód, rozczarowanie, wątpliwość, niepewność, powiątpiewanie, dół, dołek, spadły, spaść, "
    "wytrącać, wybijać, rozchwiać, spowolnienie, zwolnienie, opadanie, opadać, obniżać, zawieść, rozczarować, upadać, "
    "padać, ryzykować , ryzykowanie, narażać, ryzyko, niewiele, niedużo, mało, mniejszość, mniejszy, niski, "
    "niewysoki, depresja, nisko, niewysoko, mało, obniżać, zniżać, obcinać, zmniejszaćm, drobny, nieważny, mały, "
    "przeczenie, kwestionowanie, kara, sankcja, grzywna, rygor, odmówić, odrzucić, upadek, niepowodzenie, koniec, "
    "mały, niski, drobny, wydać, wyłożyć, wydatkować, groźba, niebezpieczeństwo, zagrożenie, kłopot, tarapaty, "
    "przeboje, problemy, niepewny, spekulacyjny, niewiarygodny, niekorzystny, zły, nieopłacalny, nierentowny, "
    "niekorzyść, szwank, strata, słaby, osłabły, wątły, nietrwały, osłabienie, podkopanie, łagodzenie, słabość, "
    "nietrwałość, wątłość, niesolidność, pogarszać, psuć, ubożyć")

if process_texts:
    print("Reading text files from " + source_data + "...")
    file_index = []
    # Traverse all sub-directories of corpus and find all files - make sure all files are text not binary
    for root, dirs, files in os.walk(source_data):
        for file in files:
            file_index.append(os.path.join(root, file))
    texts_raw = read_texts(file_index, text_encoding)
    print("Starting genismmer/corpuser processing for corpus: " + corpus_name)
    texts = gensimmer(corpuser(texts_raw))
    print("Done pre-processing %d texts." % len(texts))
    dictionary = corpora.Dictionary(texts)
    print('Number of unique tokens in raw dictionary: %d' % len(dictionary))
    if filter_extremes:
        dictionary.filter_extremes(no_below=3, no_above=0.8)  # Filter out rare and very common words
        print('Number of unique tokens after filtering out extremes: %d' % len(dictionary))
    if filter_stoplist:
        stoplist_iterable = make_stoplist_iterable("stop_lista_kombo_utf8.txt", "utf-8")
        for w in ['bogdanka', 'enea', 'zrew', 'gtc', 'mln', 'mpln', 'pkd', 'citi', 'ności', 'polimex', 'fbserwis',
                  'eurocosh', 'mbanku', 'multisport', 'simulator', 'cyberpunk', 'rozidęzań', 'helios', 'reserved',
                  'pgnig', 'with', 'proc', 'strategic']:
            # add here whatever you need to exclude
            stoplist_iterable.append(w)
        stoplist_ids = []
        for w in stoplist_iterable:
            try:
                bad_id = dictionary.token2id[w]
            except KeyError:
                continue
            stoplist_ids.append(bad_id)
        dictionary.filter_tokens(bad_ids=stoplist_ids)
        print('Number of unique tokens after filtering for stoplist: %d' % len(dictionary))
    dictionary.compactify()  # remove gaps
    with open("./data/" + corpus_name + 'texts_processed' + '.pk', 'wb') as fl:
        pickle.dump(texts, fl)
    with open("./data/" + corpus_name + 'dictionary' + '.pk', 'wb') as fl:
        pickle.dump(dictionary, fl)
    with open("./data/" + corpus_name + 'file_index' + '.pk', 'wb') as fl:
        pickle.dump(file_index, fl)
    print('Done processing. Texts and dictionary dumped to files.')
    del texts_raw

if corpus_load:
    with open("./data/" + corpus_name + 'texts_processed' + '.pk', 'rb') as fl:
        texts = pickle.load(fl)
    with open("./data/" + corpus_name + 'dictionary' + '.pk', 'rb') as fl:
        dictionary = pickle.load(fl)
    with open("./data/" + corpus_name + 'file_index' + '.pk', 'rb') as fl:
        file_index = pickle.load(fl)
    print('Processed texts loaded successfully.')

print('Creating doc2bow corpus.')
corpus = [dictionary.doc2bow(t) for t in texts]


print('Creating TFIDF model.')
tfidf_model = TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]

print('Number of documents in corpus: %d' % len(corpus))

# DATAFRAMES #
dataframes = dict()

# SCORING #
dataframes['HenryPosScore'] = corpus_scoring(word_lists['HenryPos'], dictionary)

# WORDCLOUDS #
if make_wordclouds:
    print("Making BOW wordcloud for corpus")
    fig = plt.figure()
    wc = WordCloud(background_color='white', width=3000, height=2000, color_func=black_color_func,
                   stopwords=set({'bogdanka', 'enea', 'zrew', 'gtc', 'mln', 'pkd'})).generate_from_frequencies(
        frequencies=freq_from_gensim(dictionary))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Wordcloud for corpus " + corpus_name)
    # plt.show()  # pops up the graph while running script
    plt.savefig(
        os.path.join('./results/' + corpus_name + '_wordcloud_BOW' + '.png'),
        format='png',
        dpi=300)
    plt.close(fig)

    print('Making TFIDF wordclouds.')
    # useful example: https://notebook.community/aerymilts/comparative-tweets/comparative-tweets
    fig = plt.figure()
    wc = WordCloud(background_color='white', width=3000, height=2000, color_func=black_color_func,
                   stopwords=set({'bogdanka', 'enea', 'zrew', 'gtc', 'mln', 'mpln', 'pkd', 'citi', 'ności', 'polimex', 'fbserwis', 'eurocosh', 'mbanku', 'multisport', 'simulator', 'cyberpunk', 'rozidęzań', 'helios', 'reserved'})).generate_from_frequencies(
        frequencies=tfidf_token2id(tfidf_corpus, dictionary))
    # save csv file with frequencies for use with http://wordclouds.com
    freq = tfidf_token2id(tfidf_corpus, dictionary)
    pd.DataFrame({"words": freq.keys(), "weights": freq.values()}).to_csv(
        os.path.join('./results/' + corpus_name + '_wordcloud_TFIDF' + '.csv'))
    del freq
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Wordcloud for corpus " + corpus_name)
    # plt.show()  # pops up the graph while running script
    plt.savefig(
        os.path.join('./results/' + corpus_name + '_wordcloud_TFIDF' + '.png'),
        format='png',
        dpi=300)
    plt.close(fig)

    print('Making BOW wordcloud dataframe from HenryPos')
    dataframes['HenryPos'] = wordcloud_from_list(word_lists['HenryPos'], dictionary, "Henry Positive")

    print('Making BOW wordcloud dataframe from HenryNeg')
    dataframes['HenryNeg'] = wordcloud_from_list(word_lists['HenryNeg'], dictionary, "Henry Negative")

# ### Detailed Corpus Analysis ### #
if corpus_analysis:
    print("Analyzing corpus to obtain detailed scores, both BOW counts and TFIDF.")
    scores: Dict = {}

    def wordlist_scoring(gensim_dictionary, wordlist):
        output: Dict = {}
        gensim_ids = get_gensim_ids(wordlist, gensim_dictionary)
        score_list = []
        score_list_tfidf = []
        # simple counts
        for document in corpus:
            hits = 0
            for tpl in document:
                if tpl[0] in gensim_ids:
                    hits += tpl[1]
            score_list.append(hits)
        # TFIDF frequencies
        for document in tfidf_corpus:
            hits = 0
            for tpl in document:
                if tpl[0] in gensim_ids:
                    hits += tpl[1]
            score_list_tfidf.append(hits)
        # output to dictionary
        output = {'COUNTS': score_list, 'TFIDF': score_list_tfidf}
        return output

    scores['wordcount'] = [sum([count for word,count in doc]) for doc in corpus]
    scores['HenryPos'] = wordlist_scoring(dictionary, word_lists['HenryPos'])
    scores['HenryNeg'] = wordlist_scoring(dictionary, word_lists['HenryNeg'])
    # Print descriptive statistics for scores
    descriptiveStatistics("Henry Positive BOW", scores['HenryPos']['COUNTS'])
    descriptiveStatistics("Henry Negative BOW", scores['HenryNeg']['COUNTS'])
    descriptiveStatistics("Henry Positive TFIDF", scores['HenryPos']['TFIDF'])
    descriptiveStatistics("Henry Negative TFIDF", scores['HenryNeg']['TFIDF'])

    # Organize results into dataframes
    dataframes["doc_scores_BOW"] = pd.DataFrame({"Doc_name": file_index, "WordCount": scores['wordcount'],
                                             "HenryPosBOW": scores['HenryPos']['COUNTS'],
                                             "HenryNegBOW": scores['HenryNeg']['COUNTS'],
                                             })
    dataframes["doc_scores_TFIDF"] = pd.DataFrame({"Doc_name": file_index, "WordCount": scores['wordcount'],
                                                 "HenryPosTFIDF": scores['HenryPos']['TFIDF'],
                                                 "HenryNegTFIDF": scores['HenryNeg']['TFIDF']
                                                 })
    dataframes["doc_scores_BOW"]["HenryNetPosBOW"] = (dataframes["doc_scores_BOW"]["HenryPosBOW"] - dataframes["doc_scores_BOW"]["HenryNegBOW"]) / (dataframes["doc_scores_BOW"]["HenryPosBOW"] + dataframes["doc_scores_BOW"]["HenryNegBOW"])
    dataframes["doc_scores_TFIDF"]["HenryNetPosTFIDF"] = (dataframes["doc_scores_TFIDF"]["HenryPosTFIDF"] - dataframes["doc_scores_TFIDF"]["HenryNegTFIDF"]) / (dataframes["doc_scores_TFIDF"]["HenryPosTFIDF"] + dataframes["doc_scores_TFIDF"]["HenryNegTFIDF"])
    # Print descriptive statistics for NetPos Ratios
    descriptiveStatistics("Henry NetPositive BOW", dataframes["doc_scores_BOW"]["HenryNetPosBOW"])
    descriptiveStatistics("Henry NetPositive TFIDF", dataframes["doc_scores_TFIDF"]["HenryNetPosTFIDF"])

    dataframes["doc_scores_BOW"].to_csv(('./results/' + corpus_name + '_' + "doc_scores_BOW" + '.csv'))
    dataframes["doc_scores_TFIDF"].to_csv(('./results/' + corpus_name + '_' + "doc_scores_TFIDF" + '.csv'))

    # Dump results in dataframes to CSV
    with open("./data/" + corpus_name + '_dataframes' + '.pk', 'wb') as fl:
        pickle.dump(dataframes, fl)

if lda_modelling:
    # filter away some bad tokens that popped up after text processing
    if filter_bad_tokens:
        drop_ids = []
        for t in ['gra', 'gracz', 'growy', 'gaming', 'gamingowy', 'telewizja', 'telewizyjny', 'mebel', 'meblowy', 'złoże']:
            try:
                idt = dictionary.token2id[t]
            except:
                continue
            drop_ids.append(idt)
        dictionary.filter_tokens(bad_ids=drop_ids)
        corpus = [dictionary.doc2bow(t) for t in texts]
        dictionary.compactify()  # remove gaps
    # save the dictionary to make sure you have the same one as LDA model used
    with open("./data/" + corpus_name + 'dictionary_lda' + '.pk', 'wb') as fl:
        pickle.dump(dictionary, fl)
    # change number of topics as you see fit.
    print('Beginning gensim topic modelling.')
    for x in range(2, 7, 4):
        temp = dictionary[0]  #activate dictionary so that id2word loads
        id2word = dictionary.id2token
        num_topics = x
        chunksize = 40000
        passes = 10  # reduce to make the algorythm faster in preliminary runs.
        iterations = 100
        eval_every = None  # Don't evaluate model perplexity, takes too much time, unless you want to determine the right number of passes/iterations - check for convergence in the log.

        # Train LDA Model
        # Check in the log.txt if documents converge by the final pass (diff = small enough?).
        # You may want to set random_state fore replicability using https://www.random.org/
        print("Training LDA Model with no. of topics = " + str(num_topics))
        lda_model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every,
            random_state=4164  # need to set random state to repeat and get the same results
        )

        # Pickle the model for later use, e.g. for scoring topics in your data
        try:
            pickle.dump(lda_model,
                        open(os.path.join('./results_gensim/'+ corpus_name + '_lda_dump_' + str(num_topics) + '.pk'), 'wb'))
        except:
            os.makedirs(os.path.join('./results_gensim'))
            pickle.dump(lda_model,
                        open(os.path.join('./results_gensim/'+ corpus_name + '_lda_dump_' + str(num_topics) + '.pk'), 'wb'))

        print('The top 10 keywords in each topic')
        pprint(lda_model.print_topics(num_words=10))
        logging.debug('The top 10 keywords in each topic')
        logging.debug(pformat(lda_model.print_topics(num_words=10)))

        # CV topic coherence https://rare-technologies.com/what-is-topic-coherence/
        top_topics = lda_model.top_topics(corpus)  # , num_words=20)
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        print('Average CV topic coherence: %.4f.' % avg_topic_coherence)
        """
        # Umass topic coherence - provides the same result as the default
        # https://radimrehurek.com/gensim/models/coherencemodel.html
        cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
        umass_coherence = cm.get_coherence()  # get coherence value
        print('Umass topic coherence: %.4f.' % umass_coherence)

        # UCI topic coherence - throws errors - remember to "undelete" texts above
        # https://radimrehurek.com/gensim/models/coherencemodel.html
        cm = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_uci')
        uci_coherence = cm.get_coherence()  # get coherence value
        print('UCI topic coherence: %.4f.' % uci_coherence)
        """

        if lda_wordclouds:
            def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                """
                Create word cloud in grey scale at 300 dpi for publication
                """
                return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 95)


            for t in range(lda_model.num_topics):
                fig = plt.figure()
                wc = WordCloud(background_color='black', width=3000, height=2000, color_func=grey_color_func).fit_words(
                    dict(lda_model.show_topic(t, 200)))
                plt.imshow(wc)
                plt.axis("off")
                plt.title("Topic #" + str(t))
                # plt.show()  # pops up the graph while running script
                plt.savefig(
                    os.path.join('./results_gensim/'+ corpus_name + '_wordcloud_' + str(num_topics) + '_' + str(t) + '.png'),
                    format='png',
                    dpi=300)
                plt.close(fig)

    print("Finished gensim topic modelling, check results_gensim for visuals and log.txt for details.")

# Apply selected LDA model to get document probabilities from whatever is defined under "corpus"
# see get_document_topics at https://radimrehurek.com/gensim/models/ldamodel.html
# The trick is that the model only looks at tuples of word_ids and frequencies, not words.
# more on inference here https://miningthedetails.com/LDA_Inference_Book/lda-inference.html
if forecast_probabilities:
    print("Forecasting document probabilities on the same corpus, assuming LDA modelling was done...")
    num_topics = 6  # change as needed
    try:
        lda_model = pickle.load(open(os.path.join('./results_gensim/'+ corpus_name + '_lda_dump_' + str(num_topics) + '.pk'), 'rb'))
    except:
        print("LDA model file not found, exiting.")
        exit(1)
    lda_topic_list = []
    for d in corpus:
        doc_topics = lda_model.get_document_topics(d)
        temp_dict = {}
        for t in doc_topics:
            temp_dict[t[0]] = [t[1]]
        lda_topic_list.append(temp_dict)
    df_doc_topics = pd.concat([pd.DataFrame.from_dict(doct) for doct in lda_topic_list], axis=0, ignore_index=True)
    # print summary stats and save to CSV
    for c in df_doc_topics.columns:
        # df_doc_topics[c][np.isnan(df_doc_topics[c])] = 0  # this removes NAN to zero, for easier analysis later
        print("Descriptives for topic " + str(c))
        pprint(df_doc_topics[c].describe())
    df_doc_topics.to_csv('./results/' + corpus_name + '_lda_topics_per_doc_' + str(num_topics) + '.csv')

if forecast_probabilities_sd:
    print("Forecasting document probabilities for management reports (SD)...")
    # Create BOW corpora for each model using the model's dictionary
    try:
        with open("./data/sd_forecastsdictionary_lda.pk", 'rb') as fl:
            dict_forecasts = pickle.load(fl)
        with open("./data/sd_salesdictionary_lda.pk", 'rb') as fl:
            dict_sales = pickle.load(fl)
        with open("./data/sd_operationsdictionary_lda.pk", 'rb') as fl:
            dict_operations = pickle.load(fl)
    except:
        print("Error loading dictionaries for thematic LDA corpora. Make sure they exist.")
        exit(1)
    corpus_forecasts = [dict_forecasts.doc2bow(t) for t in texts]
    corpus_sales = [dict_sales.doc2bow(t) for t in texts]
    corpus_operations = [dict_operations.doc2bow(t) for t in texts]

    # FORECASTS
    print("Forecasting document probabilities for LDA Forecasts 11 topics:")
    num_topics = 11  # change as needed
    lda_model = pickle.load(open('./results_gensim/sd_forecasts_lda_dump_11.pk', 'rb'))
    #lda_model = pickle.load(open(os.path.join('./results_gensim/'+ corpus_name + '_lda_dump_' + str(num_topics) + '.pk'), 'rb'))
    lda_topic_list = []
    for d in corpus_forecasts:
        doc_topics = lda_model.get_document_topics(d)
        temp_dict = {}
        for t in doc_topics:
            temp_dict[t[0]] = [t[1]]
        lda_topic_list.append(temp_dict)
    df_doc_topics = pd.concat([pd.DataFrame.from_dict(doct) for doct in lda_topic_list], axis=0, ignore_index=True)
    # print summary stats and save to CSV
    for c in df_doc_topics.columns:
        # df_doc_topics[c][np.isnan(df_doc_topics[c])] = 0  # this removes NAN to zero, for easier analysis later
        print("Descriptives for topic " + str(c))
        pprint(df_doc_topics[c].describe())
    df_doc_topics.to_csv('./results/' + corpus_name + "_forecasts" + '_lda_topics_per_doc_' + str(num_topics) + '.csv')

    # SALES
    print("Forecasting document probabilities for LDA sales 2 topics:")
    num_topics = 2  # change as needed
    lda_model = pickle.load(open('./results_gensim/sd_sales_lda_dump_2.pk', 'rb'))
    # lda_model = pickle.load(open(os.path.join('./results_gensim/'+ corpus_name + '_lda_dump_' + str(num_topics) + '.pk'), 'rb'))
    lda_topic_list = []
    for d in corpus_sales:
        doc_topics = lda_model.get_document_topics(d)
        temp_dict = {}
        for t in doc_topics:
            temp_dict[t[0]] = [t[1]]
        lda_topic_list.append(temp_dict)
    df_doc_topics = pd.concat([pd.DataFrame.from_dict(doct) for doct in lda_topic_list], axis=0, ignore_index=True)
    # print summary stats and save to CSV
    for c in df_doc_topics.columns:
        # df_doc_topics[c][np.isnan(df_doc_topics[c])] = 0  # this removes NAN to zero, for easier analysis later
        print("Descriptives for topic " + str(c))
        pprint(df_doc_topics[c].describe())
    df_doc_topics.to_csv('./results/' + corpus_name + "_sales" + '_lda_topics_per_doc_' + str(num_topics) + '.csv')

    print("Forecasting document probabilities for LDA sales 6 topics:")
    num_topics = 6  # change as needed
    lda_model = pickle.load(open('./results_gensim/sd_sales_lda_dump_6.pk', 'rb'))
    # lda_model = pickle.load(open(os.path.join('./results_gensim/'+ corpus_name + '_lda_dump_' + str(num_topics) + '.pk'), 'rb'))
    lda_topic_list = []
    for d in corpus_sales:
        doc_topics = lda_model.get_document_topics(d)
        temp_dict = {}
        for t in doc_topics:
            temp_dict[t[0]] = [t[1]]
        lda_topic_list.append(temp_dict)
    df_doc_topics = pd.concat([pd.DataFrame.from_dict(doct) for doct in lda_topic_list], axis=0, ignore_index=True)
    # print summary stats and save to CSV
    for c in df_doc_topics.columns:
        # df_doc_topics[c][np.isnan(df_doc_topics[c])] = 0  # this removes NAN to zero, for easier analysis later
        print("Descriptives for topic " + str(c))
        pprint(df_doc_topics[c].describe())
    df_doc_topics.to_csv('./results/' + corpus_name + "_sales" + '_lda_topics_per_doc_' + str(num_topics) + '.csv')

    #OPERATIONS
    print("Forecasting document probabilities for LDA operations 11 topics:")
    num_topics = 11  # change as needed
    lda_model = pickle.load(open('./results_gensim/sd_operations_lda_dump_11.pk', 'rb'))
    # lda_model = pickle.load(open(os.path.join('./results_gensim/'+ corpus_name + '_lda_dump_' + str(num_topics) + '.pk'), 'rb'))
    lda_topic_list = []
    for d in corpus_operations:
        doc_topics = lda_model.get_document_topics(d)
        temp_dict = {}
        for t in doc_topics:
            temp_dict[t[0]] = [t[1]]
        lda_topic_list.append(temp_dict)
    df_doc_topics = pd.concat([pd.DataFrame.from_dict(doct) for doct in lda_topic_list], axis=0, ignore_index=True)
    # print summary stats and save to CSV
    for c in df_doc_topics.columns:
        # df_doc_topics[c][np.isnan(df_doc_topics[c])] = 0  # this removes NAN to zero, for easier analysis later
        print("Descriptives for topic " + str(c))
        pprint(df_doc_topics[c].describe())
    df_doc_topics.to_csv('./results/' + corpus_name + "_operations" + '_lda_topics_per_doc_' + str(num_topics) + '.csv')

    """
        if df_lda_topics is None:
            df_lda_topics = pd.DataFrame.from_dict(temp_dict)
        else:
            df_lda_topics.merge(pd.DataFrame.from_dict(temp_dict), copy=False)
        del temp_dict, doc_topics
    """


# Utility function to check if there are weird documents. Use if needed
def find_docs_by_token(token):
    tokenid = dictionary.token2id[str(token)]
    print('Searching corpus for documents containing token: ' + str(token))
    for d in corpus:
        for t in d:
            if t[0]==tokenid:
                print(str(corpus.index(d)) + ": " + str([dictionary.id2token[tk[0]] for tk in d]))
                print("File: " + file_index[corpus.index(d)])
                continue
    print("Search complete for tokenid: " + str(tokenid))
    return 0

# Possible extension to eradicate documents in English but it's better to do it earlier
# In Windows Powershell use: Get-ChildItem -Recurse | Select-String "million" -List | Select Path
"""
check_English = False
if check_English:
    find_docs_by_token("million")
    find_docs_by_token("from")
"""

print("Program complete. Stop.")