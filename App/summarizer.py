import os
import pandas as pd
from string import punctuation
import nltk
import pickle
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser
from nltk.tree import ParentedTree
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet') 
nltk.download('stopwords')

# vectorizer model saved (Creation of this model is described at Notebooks/feature_engineering.ipynb).
with open('pickles\\tfidf.pickle','rb') as data:
    tfidf = pickle.load(data)


with open('pickles\\svm.pickle','rb') as data:
    svm = pickle.load(data)

def triplet_extraction (parse_tree):
    
    subject = extract_subject(parse_tree)
    predicate = extract_predicate(parse_tree)
    objects = extract_object(parse_tree)
   
    return [subject, predicate, objects]


def extract_subject (parse_tree):
    
    subject = []
    for s in parse_tree.subtrees(lambda x: x.label() == 'NP'):
        for t in s.subtrees(lambda y: y.label().startswith('NN')):
            output = t[0]
            if output != [] and output not in subject:
                subject.append(output) 
    if len(subject) != 0: return subject[0] 
    
    else: return ''

def extract_predicate (parse_tree):

    output, predicate = [],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label().startswith('VB')):
            output = t[0]
            if output != [] and output not in predicate:    
                predicate.append(output)
    if len(predicate) != 0: return predicate[-1]
    else: return ''



def extract_object (parse_tree):

    objects, output, word = [],[],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label() in ['NP','PP','ADJP']):
            if t.label() in ['NP','PP']:
                for u in t.subtrees(lambda z: z.label().startswith('NN')):
                    word = u 
            else:
                for u in t.subtrees(lambda z: z.label().startswith('JJ')):
                    word = u
            if len(word) != 0:
                output = word[0]
            if output != [] and output not in objects:
                objects.append(output)
    if len(objects) != 0: return objects[0]
    else: return ''



def text_preprocess(text):
    tokenizer  = nltk.tokenize.punkt.PunktSentenceTokenizer()
    parser = CoreNLPParser(tagtype='pos')
    sentences = tokenizer.tokenize(text);
    result = []
    for sentence in sentences:
        parse_tree, = ParentedTree.convert(list(parser.parse(sentence.split()))[0])
        result.append(triplet_extraction(parse_tree)) 
    
    return result,sentences

def stemming(triplets):
    lemmatizer = WordNetLemmatizer()
    for triplet in triplets:
        triplet[1] = lemmatizer.lemmatize(triplet[1],pos='v')
    return triplets

def process_sentences(sentences):
    lemmatizer = WordNetLemmatizer()
    to_remove = stopwords.words('english') + list(punctuation) + ["``","'s","'d","''","'"]
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokens = word_tokenize(sentence)
        tokens = filter(lambda token: token not in to_remove, tokens)
        tokens = map(lambda token: lemmatizer.lemmatize(token,pos='v'),tokens)
        processed_sentences.append(' '.join(tokens))
    return processed_sentences

def parsing(txt):
    
    # For this section give path values of java, and corenlp in your system.
    
    java_path = "C:/Program Files/Java/jdk-15.0.1/bin/java.exe"
    os.environ['JAVAHOME'] = java_path
    cn_path = "D:/Learning Material/Project/Actual work/corenlp/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar" 
    model_path= "D:/Learning Material/Project/Actual work/corenlp/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0-models.jar"
    
    server = CoreNLPServer(cn_path,model_path)
    server.start()
    output,sentences = text_preprocess(txt)
    output = stemming(output)
    triplets = []
    for x in output:
        triplets.append(' '.join(x))
    processed = process_sentences(sentences)
    server.stop()
    
    return triplets,sentences,processed

def summarize(triplets,sentences,processed_sentences,r):
    
    vect = TfidfVectorizer(encoding = 'utf-8',
                        stop_words = None,
                        lowercase = False,
                        min_df = 1,
                        max_df = 1.0,
                        norm = 'l2',
                        sublinear_tf = True)
    inp = pd.Series(triplets)
    features = vect.fit_transform(inp).todense()
    
    
    n = int(len(features)*r)
    model = KMeans(n_clusters=n, init='k-means++', max_iter=50, n_init=10)
    model.fit(features)
    clusters = model.labels_.tolist()

    sent_dict = {}
    for i,sentence in enumerate(sentences):
        sent_dict[i] = {}
        sent_dict[i]['text'] = sentence
        sent_dict[i]['processed'] = processed_sentences[i]
        sent_dict[i]['cluster'] = clusters[i]

    clus_dict = {}
    for i,value in sent_dict.items():
        if value['cluster'] not in clus_dict:
            clus_dict[value['cluster']] = []
        clus_dict[value['cluster']].append(value['processed'])
        value['idx'] = len(clus_dict[value['cluster']])-1

    max_cos_score = {}
    for i,value in clus_dict.items():
        max_cos_score[i] = {}
        max_cos_score[i]['score'] = 0
        tfidf_matrix = vect.fit_transform(value)
        cos_sim_matrix = cosine_similarity(tfidf_matrix)
        for idx,row in enumerate(cos_sim_matrix):
            sum = 0
            for col in row:
                sum += col
            if sum>=max_cos_score[i]['score']:
                max_cos_score[i]['score'] = sum
                max_cos_score[i]['idx'] = idx

    result_index = []
    for i,value in max_cos_score.items():
        cluster = i
        idx = value['idx']
        for key,val in sent_dict.items():
            if val['cluster'] == cluster and val['idx'] == idx:
                result_index.append(key)

    result_index.sort()

    summary = ''
    for idx in result_index:
        summary += sentences[idx] + ' '
        
    return summary

def transform_text(txt):
    df = pd.DataFrame([txt],columns=['Content'])
    signs = list("?!,.:;")
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    stop_words = list(stopwords.words('english'))
    df['Content_parsed'] = df['Content'].str.replace('\r',' ')
    df['Content_parsed'] = df['Content_parsed'].str.replace('\n',' ')
    df['Content_parsed'] = df['Content_parsed'].str.replace('   ',' ')
    df['Content_parsed'] = df['Content_parsed'].str.replace('  ',' ')
    df['Content_parsed'] = df['Content_parsed'].str.replace('"','')
    df['Content_parsed'] = df['Content_parsed'].str.lower()
    for sign in signs:
        df['Content_parsed'] = df['Content_parsed'].str.replace(sign,'')
    df['Content_parsed'] = df['Content_parsed'].str.replace("'s","")
    words = df['Content_parsed'][0].split(" ")
    for word in words:
        lemmatized_list.append(lemmatizer.lemmatize(word,pos='v'))
    lemmatized_text = " ".join(lemmatized_list)
    df['Content_parsed'] = lemmatized_text
    for stop_word in stop_words:
        regex_stop = r"\b" + stop_word + r"\b"
        df['Content_parsed'] = df['Content_parsed'].str.replace(regex_stop,'')
    df['Content_parsed'] = df['Content_parsed'].str.replace('\\s+',' ')
    features = tfidf.transform(df['Content_parsed']).toarray()
    return features

def get_category(code):
    cat_code = {
    0:'Business',
    1:'Entertainment',
    2:'Politics',
    3:'Sport',
    4:'Tech'}
    return cat_code[code]

def predict_cat(txt):
    for_pred = transform_text(txt)
    result = get_category(svm.predict(for_pred)[0])
    prob = svm.predict_proba(for_pred)[0].max()*100
    if(prob<65):
        result = 'other'
    
    return result,prob
