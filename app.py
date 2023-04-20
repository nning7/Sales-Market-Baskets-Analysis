# MingAnn imports
import pickle
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from fuzzywuzzy import fuzz
import pandas as pd
from flask import Flask, render_template, request
import os
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib
#Amos imports
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from keybert import KeyBERT
from keytotext import trainer, make_dataset
import cv2 #keytotext requirement
import keras
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import tensorflow
tensorflow.random.set_seed(2)
from numpy.random import seed
import numpy as np
seed(1)
import string
import re
from keras.utils import np_utils
from keras.models import load_model
# JefYCE imports
import pandas as pd
import nltk
import re
from cleantext import clean
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle as pk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import exists

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('Agg')

app = Flask(__name__)
global_food_name = None


# MING ANN WORKS vv ------------------------------------------------------------------------------------------------------------------------------------------------------

def load_results():
    with open('results.pickle', 'rb') as file:
        my_dict = pickle.load(file)
    return my_dict


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has a file part
        if 'marketing_csv' not in request.files:
            return render_template('recommended_sale.html', error='No file part')

        file = request.files['marketing_csv']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('recommended_sale.html', error='No selected file')

        if file:
            filename = 'salesData.csv'
            file.save(filename)
            analysis()
            # return a message indicating success
            return render_template('recommended_sale.html', message='File uploaded successfully')

    return render_template('recommended_sale.html')


@app.route('/recommend')
def recommend():
    return render_template('recommended_sale.html')


@app.route('/up_cross', methods=['GET', 'POST'])
def up_cross():
    df = pd.read_csv('salesData.csv')
    # assume your data frame is called 'df' and contains a column called 'food_name'
    food_list = df['name'].unique().tolist()
    return render_template('up_cross.html', food_list=food_list)


def load_data():
    df = pd.read_csv('salesData.csv')
    df["orderDateTime"] = pd.to_datetime(df["orderDateTime"])
    df = df.drop_duplicates(subset='orderId')
    return df


@app.route('/plot', methods=['GET', 'POST'])
def plot():
    # food_name = request.args.get('food_name')
    global global_food_name
    if request.method == 'POST':
        # Get the food name from the form
        food_name = request.form['food_name']
        food_name = request.form.get('food_name')
        global_food_name = food_name

    graph_type = request.args.get('graph_type')
    if graph_type is None:
        graph_type = 'daily'
    # if food_name is None:
    # return 'Please enter a food name'
    df = load_data()
    food_sales = df[df['name'] == global_food_name]
    if food_sales.empty:
        return 'No sales data for this food item'
    fig, ax = plt.subplots()
    if graph_type == 'daily':
        daily_sales = food_sales.groupby(df['orderDateTime'].dt.date)[
            'totalAmount'].sum().plot(ax=ax)
        ax.set_title('Daily Sales')
        ax.set_xlabel('Day')
        ax.set_ylabel('Salas')
        plt.savefig('static/day_plot.png')
    elif graph_type == 'weekly':
        weekly_sales = food_sales.groupby(df['orderDateTime'].dt.isocalendar().week)[
            'totalAmount'].sum().plot(ax=ax)
        ax.set_title('Weekly Sales')
        ax.set_xlabel('Week')
        ax.set_ylabel('Salas')
        plt.savefig('static/week_plot.png')
    elif graph_type == 'monthly':
        monthly_sales = food_sales.groupby(df['orderDateTime'].dt.month)[
            'totalAmount'].sum().plot(ax=ax)
        ax.set_title('Monthly Sales')
        ax.set_xlabel('Month')
        ax.set_ylabel('Salas')
        plt.savefig('static/month_plot.png')

    plt.close(fig)
    food_list = df['name'].unique().tolist()
    return render_template('up_cross.html', graph_type=graph_type, food_list=food_list, name=global_food_name)


@app.route('/campaign', methods=['GET', 'POST'])
def campaign():
    my_dict = load_results()
    df = pd.read_csv('salesData.csv')
    if request.method == 'POST':
        # Get selected key from dropdown
        selected_key = request.form.get('key')
        # Get value of selected key from dictionary
        selected_value = my_dict[selected_key]
        data = []
        data2 = []
        for i, tpl in enumerate(selected_value):
            increment = i + 1
            first = tpl[0]
            rest = tpl[1:]
            data.append((increment, first, rest))
            increment2 = 0
        increment2 = 0
        for item in rest:
            # get the price of the food from the sales_data DataFrame

            price = df.query("name == @item")['amount'].values[0]
            # compare prices and add to list
            # get the price of the food in list1, or set it to 0 if it is not in list1
            list1_price = df.query("name == @first")['amount'].values[0]

            # compare the prices of the two foods
            if price > list1_price:
                increment2 = increment2 + 1
                data2.append((increment2, first, rest))

        return render_template('market_product.html', my_dict=my_dict, selected_key=selected_key, data1=data, data2=data2)
    else:
        # Render template with dropdown populated with dictionary keys
        return render_template('market_product.html', my_dict=my_dict, selected_key=None, selected_value=None)


@app.route('/dict')
def dict():
    # Load the dictionary from the pickle file
    with open('my_dict.pickle', 'rb') as f:
        my_dict = pickle.load(f)

    df = pd.DataFrame(list(my_dict.items()), columns=[
                      'Items', 'Lists of Products Being Referenced'])
    table = df.to_html(index=False)
    return render_template('dict.html', table=table)

    # Render the template with the dictionary contents
    # return render_template('dict.html', dict_contents=dict_contents)


def analysis():
    df = pd.read_csv('salesData.csv')
    food_list = df['name'].unique().tolist()
    grs = list()  # groups of names with distance > 80
    for name in food_list:
        for g in grs:
            if all(fuzz.token_sort_ratio(name, w) > 80 for w in g):
                g.append(name)
                break
        else:
            grs.append([name, ])

    my_dict = {}
    for sublist in grs:
        key = sublist[0]
        value = sublist
        my_dict[key] = value

    # Open a file in binary mode and save the dictionary as a pickle file
    with open('my_dict.pickle', 'wb') as f:
        pickle.dump(my_dict, f)
    df1 = df
    # Loop through each value in the DataFrame
    for i, row in df.iterrows():
        value = row['name']
        for lst in grs:
            if value in lst:
                df.loc[i, 'name'] = lst[0]
                break  # Stop searching after finding the first matching list

    basket = (df.groupby(['orderId', 'name'])['quantity'].sum(
    ).unstack().reset_index().fillna(0).set_index('orderId'))
    encode_basket = basket.applymap(encode_units)
    filter_basket = encode_basket[(encode_basket > 0).sum(axis=1) >= 2]
    res = fpgrowth(filter_basket, min_support=0.01,
                   use_colnames=True).sort_values('support', ascending=False)
    res = association_rules(res, metric="lift", min_threshold=1).sort_values(
        'lift', ascending=False).reset_index(drop=True)
    res['food_pairs'] = [frozenset.union(
        *X) for X in res[['antecedents', 'consequents']].values]
    # Convert tuples to frozensets so we can sort them
    res['food_pairs'] = res['food_pairs'].apply(lambda x: frozenset(sorted(x)))

    # Drop duplicates, including those with reversed tuples
    res = res.drop_duplicates(subset=['food_pairs'])

    # Convert frozensets back to tuples
    res['food_pairs'] = res['food_pairs'].apply(lambda x: tuple(sorted(x)))

    # Convert the frozenset column to a tuple
    food_pairs = res['food_pairs'].apply(lambda x: tuple(x))
    food_pairs = food_pairs.drop_duplicates()

    # Group the foods by shop
    grouped = df.groupby('name.1')['name'].apply(list)

    # Initialize an empty dictionary to store the results
    results = {}

    # Iterate over the shops
    for shop, foods in grouped.items():
        # Check if any of the food pairs are in the shop
        pairs = [pair for pair in food_pairs if all(
            food in foods for food in pair)]
        # If there are any pairs, add them to the results dictionary
        if pairs:
            results[shop] = pairs

    # group food names by shop into a dictionary
    original_name = {}
    for shop, group in df1.groupby('name.1'):
        original_name[shop] = list(group['name'])

    for shop, items in results.items():
        for i, item in enumerate(items):
            new_item = list(item)
            for j, food in enumerate(item):
                if food in my_dict:
                    original_food_name = original_name[shop][j]
                    new_food_name = my_dict[food][0]
                    if new_food_name in original_name[shop]:
                        new_item[j] = new_food_name
                    else:
                        new_item[j] = original_food_name
            items[i] = tuple(new_item)
        results[shop] = items

    with open('results.pickle', 'wb') as f:
        pickle.dump(results, f)


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

# MING ANN WORKS ^^ ------------------------------------------------------------------------------------------------------------------------------------------------------


# JEFRY WORKS vv ---------------------------------------------------------------------------------------------------------------------------------------------------------

# Directories of marketing_topic.html

@app.route('/marketing_topic')
def marketing_topic():
    return render_template('marketing_topic.html')

@app.route('/uploadCSV', methods=['GET', 'POST'])
def uploadFacebookCSV():
    if request.method == 'POST':
        # check if the post request has a file part
        if 'marketing_csv' not in request.files:
            return render_template('marketing_topic.html', error='No file selected')

        file = request.files['marketing_csv']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('marketing_topic.html', error='No file selected')
        elif file:
            filename = 'marketing_topic.csv'
            file.save(filename)
            df = pd.read_csv('marketing_topic.csv', encoding='cp1252')
            df.to_csv('marketing_topic.csv',index=False)
            dataprocessing()
            # return a message indicating success
            return render_template('marketing_topic.html', message='File uploaded successfully')

    return render_template('marketing_topic.html')

@app.route('/hot_marketing', methods=['GET', 'POST'])
def hot_marketing_topic():
    file = 'clean_dataset.csv'
    file_exists = exists(file)
    if file_exists:
        df = pd.read_csv(file)
    else:
        return render_template('hot_marketing_topic.html', notfound='CSV not found, please upload CSV')
    
    # vectorizer = CountVectorizer(max_df=1.0, min_df=0, token_pattern='\w+|\$[\d\.]+|\S+')
    vectorizer = CountVectorizer(token_pattern='\w+|\$[\d\.]+|\S+', stop_words='english')
    
    tf = vectorizer.fit_transform(df['message'].values.astype('U'))
    tf_feature_names = vectorizer.get_feature_names_out()
    number_of_topics = 5
    model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
    model.fit(tf)
    def display_topics(model, feature_names, no_top_words):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
        return pd.DataFrame(topic_dict)
    no_top_words = 20
    result = display_topics(model, tf_feature_names, no_top_words)
    header = result.columns.tolist()
    data = []
    for i in result.index:
        data.append(result.iloc[i])
    return render_template('hot_marketing_topic.html', header=header, data=data)
    
@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment():
    pathname1 = "clean_dataset.csv"
    pathname2 = "filtered_dataset.csv"
    
    file_exists1 = exists(pathname1)
    file_exists2 = exists(pathname2)
    if file_exists1 and file_exists2:
        df1 = pd.read_csv(pathname1)
        df = pd.read_csv(pathname2)
    else:
        return render_template('sentiment_analysis.html', notfound='CSV not found, please upload CSV')

    df['message'] = df1['message']
    # df = df.drop(['Unnamed: 0', 
    #               'care',
    #               'total',
    #               'comments',
    #               'shares'], axis=1)
    col = ['like', 'love', 'haha', 'wow', 'sad', 'angry']
    df[col] = df[col].astype(int)
    
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        scores = analyzer.polarity_scores(str(text))
        sentiment = 1 if scores['pos'] > 0 else 0
        return sentiment

    df['sentiment'] = df['message'].apply(get_sentiment)
    header = df.columns.tolist()
    data = []
    for i in df.index[:50]:
        data.append(df.iloc[i])
    return render_template('sentiment_analysis.html', header=header, data=data)

@app.route('/reaction_prediction', methods=['GET', 'POST'])
def reaction_page():
    return render_template('reaction_prediction.html')

@app.route('/predict_reaction', methods=['GET', 'POST'])
def reaction():
    with open('reaction_model.pkl', 'rb') as f:
        prediction_model = pickle.load(f)
        
    with open('rating.pkl', 'rb') as f:
        rating_model = pickle.load(f)
    
    #get input
    sentence = request.form['sentence']
    if sentence:
        data = [sentence]
    else:
        return render_template('reaction_prediction.html', error="Please insert marketing text")
    
    #vectorizer = TfidfVectorizer(stop_words=None)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)

    # Visualizing the Document Term Matrix using TF-IDF
    VectorizedText=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    ori = pd.DataFrame(0.0, index=np.arange(len(data)), columns=prediction_model.feature_names_in_)
    for i in VectorizedText:
        for x in ori: 
            if i == x:
                ori[x] = ori[x].replace(ori[x].values, VectorizedText[i].values)
                
    prediction_result = pd.DataFrame(prediction_model.predict(ori), columns=['LIKE(01)',
                                                                             'LOVE(02)',
                                                                             'HAHA(03)',
                                                                             'WOW(04)', 
                                                                             'SAD(05)', 
                                                                             'ANGRY(06)'])
    
    # rating
    test_data = pd.concat([ori, prediction_result], axis=1)
    rating_result = rating_model.predict(test_data)

    prediction_result = prediction_result.astype(int)
    
    data = []
    for i in prediction_result.index[:1]:
        data.append(abs(prediction_result.iloc[i]))
    return render_template('reaction_prediction.html', sentence=sentence, data=data, rating="{:.4f}".format(rating_result[0][0]))
    
def dataprocessing():
    df = pd.read_csv('marketing_topic.csv')
    
    # drop null
    df.dropna(inplace= True)
    
    df1 = df.drop([ 'care',
              'total',
              'comments',
              'shares'], axis=1)
    
    # export filtered null data to csv
    df1.to_csv('filtered_dataset.csv', index=False)
    
    # clear non_usable data text
    def no_mention(df):
        return re.sub("@[A-Za-z0-9_]+","", df)

    def no_hashtag(df):
        return re.sub("#[A-Za-z0-9_]+","", df)
    

    def cleaning(df) :
        return clean(
            df,
            fix_unicode=False,
            to_ascii=True,
            lower=True,
            normalize_whitespace=True,
            no_line_breaks=False,
            strip_lines=True,
            keep_two_line_breaks=False,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=True,
            no_digits=True,
            no_currency_symbols=True,
            no_punct=True,
            no_emoji=True,
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="",
            replace_with_currency_symbol="",
            replace_with_punct="",
            lang="en"
        )

    stop_words = stopwords.words('english')
    nlp = spacy.load('en_core_web_sm')

    def lemmatize_text(text):
        return [token.lemma_ for token in nlp(text)]

    def tokenize_sentences(df):
        sentences = nltk.sent_tokenize(df)
        return sentences
    
    def clean_stopwords(text):
        return lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])

    df['message'] = df['message'].apply(no_mention)
    df['message'] = df['message'].apply(no_hashtag)
    df['message'] = df['message'].str.strip()
    df = df['message'].str.replace('\d+', '',regex=True)
    df = df.dropna()
    df = df.apply(tokenize_sentences)
    clean_df = df.apply(cleaning)
    df = pd.DataFrame(clean_df)
    #df = df.dropna()
    # save preprocessed message to csv
    df.to_csv('clean_dataset.csv',index=False)

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

    
# JEFRY WORKS ^^ ---------------------------------------------------------------------------------------------------------------------------------------------------------


# AMOS WORKS OPT ----------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/ai_text')
def ai_text():
    return render_template('ai_text.html')

@app.route('/opt')
def opt():
    return render_template('opt.html')

@app.route('/opt_clean', methods=['GET', 'POST'])
def opt_clean():
    if request.method == 'POST':
        #load model from local
        PATH = 'FoodAds_OPT350m_clean_eng'
        tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(PATH, local_files_only=True)
        #get input
        sentence = request.form['sentence']
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
        generated_text = model.generate(input_ids=input_ids, max_length=50)
        output_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
        return render_template('opt.html', output_text=output_text)
    else:
        return render_template('opt.html')

@app.route('/upload_opt', methods=['GET', 'POST'])
def upload_opt():
    if request.method == 'POST':
        # check if the post request has a file part
        if 'opt.csv' not in request.files:
            return render_template('opt.html', error='No file part')
        file = request.files['opt.csv']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('opt.html', error='No selected file')
        
        if file.filename != 'opt.csv':
            return render_template('opt.html', error='Invalid File Name')

        if file:
            filename = 'opt.csv'
            file.save(filename)
            opt_retrain()
            # return a message indicating success
            return render_template('opt.html', message='File uploaded successfully')

    return render_template('opt.html')    

@app.route('/opt_retrain', methods=['GET', 'POST'])
def opt_retrain():
    df = pd.read_csv('opt.csv', encoding='utf-8')
    #preprocess
    def no_mention(text):
        return re.sub("@[A-Za-z0-9_]+","", text)
    def no_hashtag(text):
        return re.sub("#[A-Za-z0-9_]+","", text)

    df['message'] = df['message'].apply(no_mention)
    df['message'] = df['message'].apply(no_hashtag)
    #removing blank space
    df['message'] = df['message'].str.strip()
    #removing number
    df = df['message'].str.replace('\d+', '',regex=True)
    clean_df = df.dropna()
    def tokenize_sentences(df):
        sentences = nltk.sent_tokenize(df)
        return sentences
    df = df.apply(tokenize_sentences)
    def cleaning(df) :
        return clean(
            df,
            fix_unicode=False,
            to_ascii=False,
            lower=True,
            normalize_whitespace=True,
            no_line_breaks=False,
            strip_lines=True,
            keep_two_line_breaks=False,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=True,
            no_emoji=True,
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="",
            replace_with_currency_symbol="",
            replace_with_punct="",
            lang="en"
    )
    clean_df = df.apply(cleaning)
    df = pd.DataFrame(clean_df)
    #df = df.dropna()
    
    #start retrain
    PATH = 'FoodAds_OPT350m_clean_eng'
    tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(PATH, local_files_only=True)
    df = df.rename(columns={df.columns[0]: "message"})
    training_data, testing_data = train_test_split(df, test_size=0.3, random_state=42)
    train_size = training_data.shape[0] #storing numbers of training data
    test_size = testing_data.shape[0] #storing numbers of testing data
    data = (load_dataset("csv", data_files="opt.csv",split='train').train_test_split(train_size = train_size, test_size = test_size))
    model = AutoModelForCausalLM.from_pretrained(PATH, local_files_only=True)
    
    def tokenize_function350m(examples):
        tokenizer350m = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
        return tokenizer350m(examples["message"])
    tokenized350m = data.map(tokenize_function350m, batched=True, num_proc=4, remove_columns=["message"])
    
    def group_texts(examples):
        block_size = 128
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_datasets350m = tokenized350m.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    tokenizer350m = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
    #this is saved as a new model
    repo_name = "OPT_model_retrain"
    training_args = TrainingArguments(
        output_dir=repo_name,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        push_to_hub=False,
        optim="adamw_torch"
    )
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=lm_datasets350m['train'],
       eval_dataset=lm_datasets350m['test'],
       tokenizer=tokenizer350m
    )
    trainer.train()
    text = 'Training Completed!'
    return render_template('opt.html', text=text)

@app.route('/opt_retrain_deploy', methods=['GET', 'POST'])
def opt_retrain_deploy():
    if request.method == 'POST':
        #load model from local
        PATH = 'OPT_model_retrain'
        tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(PATH, local_files_only=True)
        #get input
        sentence = request.form['sentence']
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
        generated_text = model.generate(input_ids=input_ids, max_length=50)
        output_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
        return render_template('opt.html', output_text=output_text)
    else:
        return render_template('opt.html')

# AMOS WORKS  LSTM----------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/lstm')
def lstm():
    return render_template('lstm.html')

@app.route('/lstm_lemma', methods=['GET', 'POST'])
def lstm_lemma():
    def get_sequence_of_tokens(df_list):
        tokenizer.fit_on_texts(df_list)
        total_words = len(tokenizer.word_index) + 1
        input_sequences = []
        for line in df_list:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        return input_sequences, total_words

    def generate_padded_sequences(input_sequences):
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        label = keras.utils.np_utils.to_categorical(label, num_classes=total_words)
        return predictors, label, max_sequence_len

    def create_model(max_sequence_len, total_words):
        input_len = max_sequence_len - 1
        model = Sequential()
        model.add(Embedding(total_words, 10, input_length=input_len))
        model.add(LSTM(50))
        model.add(Dropout(0.1))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model
    
    def generate_text(seed_text, next_words, model, max_sequence_len):
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predict_x = model.predict(token_list)
            classes_x = np.argmax(predict_x,axis=1)

            output_word = ""
            for word,index in tokenizer.word_index.items():
                if index == classes_x:
                    output_word = word
                    break
            seed_text += " "+output_word
        return seed_text.title()
    
    df = pd.read_csv("eng_lemma.csv")
    df = df.astype(str)
    df_list = df['message'].values.tolist()
    df_list = [s for s in df_list if 10 <= len(s.split()) <= 50]
    df = pd.DataFrame (df_list, columns = ['message'])
    df_list = df['message'].values.tolist()
    tokenizer = Tokenizer()
    
    inp_sequences, total_words = get_sequence_of_tokens(df_list)
    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
    model = create_model(max_sequence_len, total_words)
    model = load_model('LSTM')
    sentence = request.form['sentence']
    num_choices = request.form['num_choices']
    res = generate_text(sentence, int(num_choices), model, max_sequence_len)
    return render_template('lstm.html', res=res)

@app.route('/upload_lstm', methods=['GET', 'POST'])
def upload_lstm():
    if request.method == 'POST':
        # check if the post request has a file part
        if 'lstm.csv' not in request.files:
            return render_template('lstm.html', error='No file part')
        file = request.files['lstm.csv']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('lstm.html', error='No selected file')
        
        if file.filename != 'lstm.csv':
            return render_template('lstm.html', error='Invalid File Name')

        if file:
            filename = 'lstm.csv'
            file.save(filename)
            # calling the function of retraining
            lstm_lemma_retrain()
            return render_template('lstm.html', message='File uploaded successfully')
    return render_template('lstm.html')

@app.route('/lstm_lemma_retrain', methods=['GET', 'POST'])
def lstm_lemma_retrain():
    def get_sequence_of_tokens(df_list):
        tokenizer.fit_on_texts(df_list)
        total_words = len(tokenizer.word_index) + 1
        input_sequences = []
        for line in df_list:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        return input_sequences, total_words

    def generate_padded_sequences(input_sequences):
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        label = keras.utils.np_utils.to_categorical(label, num_classes=total_words)
        return predictors, label, max_sequence_len

    def create_model(max_sequence_len, total_words):
        input_len = max_sequence_len - 1
        model = Sequential()
        model.add(Embedding(total_words, 10, input_length=input_len))
        model.add(LSTM(50))
        model.add(Dropout(0.1))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model
    
    def generate_text(seed_text, next_words, model, max_sequence_len):
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predict_x = model.predict(token_list)
            classes_x = np.argmax(predict_x,axis=1)

            output_word = ""
            for word,index in tokenizer.word_index.items():
                if index == classes_x:
                    output_word = word
                    break
            seed_text += " "+output_word
        return seed_text.title()
    
    df1 = pd.read_csv("eng_lemma.csv")
    df2 = pd.read_csv("lstm.csv")
    first_column_name = df2.columns[0]
    df2 = df2.rename(columns={first_column_name: 'message'})
    df = pd.concat([df1, df2])
    df = df.astype(str)
    
    df_list = df['message'].values.tolist()
    #filter words between 10 to 50
    df_list = [s for s in df_list if 10 <= len(s.split()) <= 50]
    df = pd.DataFrame (df_list, columns = ['message'])
    df_list = df['message'].values.tolist()
    
    #re-training
    tokenizer = Tokenizer()
    inp_sequences, total_words = get_sequence_of_tokens(df_list)
    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
    model = create_model(max_sequence_len, total_words)
    #fix the epochs at here if desired to train
    history = model.fit(predictors, label,epochs=1, verbose=1)
    #model = model.fit(predictors, label,epochs=2, verbose=1)
    model.save('LSTM_lemma_retrain')
    return render_template('lstm.html')

@app.route('/lstm_retrain_deploy', methods=['GET', 'POST'])
def lstm_retrain_deploy():
    def get_sequence_of_tokens(df_list):
        tokenizer.fit_on_texts(df_list)
        total_words = len(tokenizer.word_index) + 1
        input_sequences = []
        for line in df_list:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        return input_sequences, total_words

    def generate_padded_sequences(input_sequences):
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        label = keras.utils.np_utils.to_categorical(label, num_classes=total_words)
        return predictors, label, max_sequence_len

    def create_model(max_sequence_len, total_words):
        input_len = max_sequence_len - 1
        model = Sequential()
        #model = load_model('AI Ads/LSTM/LSTM_lemma')
        model.add(Embedding(total_words, 10, input_length=input_len))
        model.add(LSTM(50))
        model.add(Dropout(0.1))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model
    
    def generate_text(seed_text, next_words, model, max_sequence_len):
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predict_x = model.predict(token_list)
            classes_x = np.argmax(predict_x,axis=1)

            output_word = ""
            for word,index in tokenizer.word_index.items():
                if index == classes_x:
                    output_word = word
                    break
            seed_text += " "+output_word
        return seed_text.title()
    
    df1 = pd.read_csv("eng_lemma.csv")
    df2 = pd.read_csv("lstm.csv")
    first_column_name = df2.columns[0]
    df2 = df2.rename(columns={first_column_name: 'message'})
    df = pd.concat([df1, df2])
    df = df.astype(str)
    
    df_list = df['message'].values.tolist()
    #filter words between 10 to 50
    df_list = [s for s in df_list if 10 <= len(s.split()) <= 50]
    df = pd.DataFrame (df_list, columns = ['message'])
    df_list = df['message'].values.tolist()
    
    #re-training
    tokenizer = Tokenizer()
    inp_sequences, total_words = get_sequence_of_tokens(df_list)
    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
    model = create_model(max_sequence_len, total_words)
    model = load_model('LSTM_lemma_retrain')
    sentence = request.form['sentence']
    num_choices = request.form['num_choices']
    res = generate_text(sentence, int(num_choices), model, max_sequence_len)
    return render_template('lstm.html', output_text=res)
           
# AMOS WORKS K2T ----------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/k2t')
def k2t():
    return render_template('keytotext.html')

@app.route('/keytotext', methods=['GET', 'POST'])
def keytotext():
    if request.method == 'POST':
        #model = pipeline("text2text-generation", model="amosc00/k2t_AI_Ads_Foods")
        PATH = 'k2t_model'
        model = pipeline("text2text-generation", model = PATH)
        #get input
        sentence = request.form['sentence']
        generated_text = model(sentence)
        # keep only text
        generated_text = generated_text[0]['generated_text']
        return render_template('keytotext.html', generated_text=generated_text)
    else:
        return render_template('keytotext.html')

@app.route('/upload_k2t', methods=['GET', 'POST'])
def upload_k2t():
    if request.method == 'POST':
        # check if the post request has a file part
        if 'k2t.csv' not in request.files:
            return render_template('keytotext.html', error='No file part')
        file = request.files['k2t.csv']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('keytotext.html', error='No selected file')
        
        if file.filename != 'k2t.csv':
            return render_template('keytotext.html', error='Invalid File Name')

        if file:
            filename = 'k2t.csv'
            file.save(filename)
            # return a message indicating success
            k2t_retrain()
            return render_template('keytotext.html', message='File uploaded successfully')
        #k2t_retrain()
    return render_template('keytotext.html')

@app.route('/k2t_retrain', methods=['GET', 'POST'])
def k2t_retrain():
    #combine old and new csv
    df1 = pd.read_csv("eng_lemma.csv") 
    df2 = pd.read_csv("k2t.csv")
    first_column_name = df2.columns[0]
    df2 = df2.rename(columns={first_column_name: 'message'})
    df = pd.concat([df1, df2])
    df.columns.values[0] = 'message'
    #preprocess
    def no_mention(text):
        return re.sub("@[A-Za-z0-9_]+","", text)
    def no_hashtag(text):
        return re.sub("#[A-Za-z0-9_]+","", text)

    df['message'] = df['message'].apply(no_mention)
    df['message'] = df['message'].apply(no_hashtag)
    #removing blank space
    df['message'] = df['message'].str.strip()
    #removing number
    df = df['message'].str.replace('\d+', '',regex=True)
    clean_df = df.dropna()
    def tokenize_sentences(df):
        sentences = nltk.sent_tokenize(df)
        return sentences
    df = df.apply(tokenize_sentences)
    def cleaning(df) :
        return clean(
            df,
            fix_unicode=False,
            to_ascii=False,
            lower=True,
            normalize_whitespace=True,
            no_line_breaks=False,
            strip_lines=True,
            keep_two_line_breaks=False,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=True,
            no_emoji=True,
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="",
            replace_with_currency_symbol="",
            replace_with_punct="",
            lang="en"
    )
    clean_df = df.apply(cleaning)
    df = pd.DataFrame(clean_df)
    #df = df.dropna()
    
    #start training
    PATH = 'k2t_model'
    tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(PATH, local_files_only=True)
    #keyword generation
    kw_model = KeyBERT(model='all-mpnet-base-v2')
    keywords_list = []
    for message in df['message']:
        keywords = kw_model.extract_keywords(message, 
                                              keyphrase_ngram_range=(1, 3), 
                                              stop_words='english', 
                                              #highlight=False,
                                              top_n=10)
        keywords_list.append(keywords)
    
    df['keywords'] = ''
    df['keywords'] = keywords_list
    df2 = df
    df2['keywords'] = df2['keywords'].astype(str)
    pattern = r'\d+'
    df2['keywords'] = df2['keywords'].apply(lambda x: re.sub(pattern, '', x))
    df2['keywords'] = df2['keywords'].apply(lambda x: x.replace("'", "").replace(",", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(".", ""))
    df3 = df2.rename(columns={'message': 'text'})
    training_data, testing_data = train_test_split(df, test_size=0.3, random_state=42)
    model = trainer()
    model.from_pretrained(model_name="t5-base")
    model.save_model()
    return render_template('keytotext.html')

@app.route('/k2t_retrain_deploy', methods=['GET', 'POST'])
def k2t_retrain_deploy():
    if request.method == 'POST':
        #load model from local
        PATH = 'model'
        model = pipeline("text2text-generation", model = PATH)
        #get input
        sentence = request.form['sentence']
        generated_text = model(sentence)
        # keep only text
        generated_text = generated_text[0]['generated_text']
        return render_template('keytotext.html', output_text=generated_text)
    else:
        return render_template('keytotext.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
