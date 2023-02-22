import os
import snowflake.connector
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from scipy.special import softmax
import psycopg2
from transformers import AutoTokenizer, AutoConfig, TFAutoModelForTokenClassification, TFAutoModel, TFAutoModelForSequenceClassification
from keras.callbacks import LearningRateScheduler
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow as tf
import glob
# load_dotenv(dotenv_path='/home/ron/.env')
load_dotenv(dotenv_path='/Users/Jacob.George/.env')
load_dotenv(dotenv_path='/home/shared/code/.env')

#database stuff
def redshift_conn():

    conn=psycopg2.connect(dbname = os.environ['REDSHIFT_DBNAME'],
            host = os.environ['REDSHIFT_HOST'],
            port = os.environ['REDSHIFT_PORT'],
            user = os.environ['REDSHIFT_USERNAME'],
            password = os.environ['REDSHIFT_PASSWORD'])
    return conn, conn.cursor()

def snowflake_conn(db = 'FUA'):
    conn = snowflake.connector.connect(
        user = os.environ[f'{db}_SNOWFLAKE_USERNAME'],
        password = os.environ[f'{db}_SNOWFLAKE_PASSWORD'],
        account=os.environ[f'{db}_SNOWFLAKE_ACCOUNT'],
        warehouse = os.environ[f'{db}_SNOWFLAKE_WAREHOUSE'],
        database = os.environ[f'{db}_SNOWFLAKE_DATABASE'],
        role = os.environ[f'{db}_SNOWFLAKE_ROLE']
        )
    return conn, conn.cursor()

def psql_table_query_df(cur, query, ucase_cols = False):
	cur.execute(query);
	res = cur.fetchall();
	if len(res) == 0:
		return pd.DataFrame([]);
		
	res = pd.DataFrame(res);
	if ucase_cols:
		res.columns = [col[0].upper() for col in cur.description];
	else:
		res.columns = [col[0].lower() for col in cur.description];
	return res

class BERTPreprocess():
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    def preprocess_inputs(self, samples):
        tokenized_samples = list(tqdm(map(self.tokenize_inputs, samples)))
        max_len = max(map(len, tokenized_samples))
        X = np.zeros((len(samples), max_len), dtype=np.int32)
        for i, sentence in enumerate(tokenized_samples):
            for j, subtoken_id in enumerate(sentence):
                X[i, j] = subtoken_id
        return X
    
    def tokenize_inputs(self, sample):
        seq = [
                (subtoken)
                for token in sample
                for subtoken in self.tokenizer(token)['input_ids'][1:-1]
            ]
        return [103] + seq + [102]
    
    def preprocess_inputs_and_labels(self,samples,schema):
        tag_index = {tag: i for i, tag in enumerate(schema)}
        tokenized_samples = list(tqdm(map(self.tokenize_inputs_and_labels, samples)))
        max_len = max(map(len, tokenized_samples))
        X = np.zeros((len(samples), max_len), dtype=np.int32)
        y = np.zeros((len(samples), max_len), dtype=np.int32)
        for i, sentence in enumerate(tokenized_samples):
            for j, (subtoken_id, tag) in enumerate(sentence):
                X[i, j] = subtoken_id
                y[i,j] = tag_index[tag]
        return X, y
    
    def tokenize_inputs_and_labels(self, sample):
        seq = [
                (subtoken, tag)
                for token, tag in sample
                for subtoken in self.tokenizer(token)['input_ids'][1:-1]
            ]
        return [(3, 'O')] + seq + [(4, 'O')]
    
    def aggregate(self, schema, sample, predictions):
        results = []
        i = 1
        item = ' '.join([s[0] for s in sample])
        for token, y_true in sample:
            nr_subtoken = len(self.tokenizer(token)['input_ids']) - 2
            pred = predictions[i:i+nr_subtoken]
            i += nr_subtoken
            y_pred = schema[np.argmax(np.mean(softmax(pred, axis = -1),axis = 0))]
            probs = np.max(np.mean(softmax(pred, axis = -1),axis = 0))
            results.append((item, token, y_true, y_pred, probs))
        return results


class ClustResample():
    def __init__(self, df,  input_col, use_pretrained_model, model_path, model_architecture, sample_n=0):
        self.df = df
        self.input_col = input_col
        self.sample_n = sample_n
        self.model_architecture = model_architecture
        self.model_path = model_path
        self.use_pretrained_model = use_pretrained_model
        
    def bert_vecs(self):
        if self.use_pretrained_model:
            model = TFAutoModel.from_pretrained(
                self.model_path, 
                output_hidden_states = True
            )
        else:
            model = TFAutoModel.from_pretrained(
                self.model_architecture, 
                output_hidden_states = True
            )
        tokenizer = AutoTokenizer.from_pretrained(self.model_architecture)
        items = self.df[self.input_col].values
        inputs = [i.split() for i in items]
        X = BERTPreprocess(tokenizer).preprocess_inputs(inputs)
        
        outputs = model(X)
        
        embeds_mean = np.array(outputs.hidden_states[-1:]).mean(axis = 0).mean(axis = 1).squeeze()
        self.df['vec'] = list(embeds_mean)
        
    def kmeans(self):
        km = KMeans(n_clusters = self.sample_n, algorithm = 'full')
        km.fit(np.array(self.df['vec'].tolist()))
        self.df['k_label'] = km.labels_
        self.df['dist_to_centroid'] = self.df.groupby(by = 'k_label')['vec'].transform(lambda x: self.compute_edist(x))
    
    def compute_edist(self, x):
        vecs = np.array(x.tolist())
        centroid = vecs.mean(axis = 0).reshape(1,-1)
        distance = euclidean_distances(centroid,vecs)
        return distance[0]
    
    def sample_centroids(self):
        self.sample = self.df.sort_values(by = 'dist_to_centroid', ascending = True).drop_duplicates(subset = 'k_label')
        self.df = self.df.drop(columns = ['vec', 'k_label', 'dist_to_centroid'])
        
    def format_for_classifier_labelling(self):
        self.out = self.sample.drop(columns = ['vec', 'k_label', 'dist_to_centroid'])
        self.out['label'] = ''
        
    def format_for_ner_labelling(self):
        out = []
        for _,row in self.sample.iterrows():
            for pos, w in enumerate(row[self.input_col].split()):
                out.append([row[self.input_col],pos, w])
        self.out = pd.DataFrame(out, columns = [self.input_col, 'position', 'word'])
        self.out = pd.merge(
            self.df[[
                self.input_col
                ]].drop_duplicates(), self.out).drop_duplicates(subset = [self.input_col, 'position', 'word'])
        self.out = self.out.sort_values(by = [self.input_col, 'position'])
        self.out['tag'] = ''

def loadPreprocess(model_name, itemname_col, training_sets, bertPreproc):
    file_list = glob.glob(f"named_entity_recognition/{model_name}/data/*_labelled.csv")
    df = pd.concat([pd.read_csv(f) for f in file_list])

    # df = pd.concat([pd.read_csv(f'named_entity_recognition/{model_name}/data/{f}').drop_duplicates() for f in training_sets])
    df['formatted_tag'] = df['tag'].str.upper()

    samples = df.groupby(by = itemname_col).apply(lambda x: [tuple(row) for row in x[['word', 'formatted_tag']].values])
    samples = samples.sample(frac = 1)
    schema = list(set(['_', 'O'] + sorted({tag for sentence in samples 
                                for _, tag in sentence})))

    X_train, Y_train = bertPreproc.preprocess_inputs_and_labels(samples,schema)
    
    return X_train, Y_train, schema

def loadTrainModel(schema, X_train, Y_train
                    ,n_epochs=100
                    ,batch_size=16
                    ,patience=5
                    ,learning_rate = 0.0001
                    ,use_pretrained_model=False
                    ,model_path =''
                    ,model_architecture='bert-base-uncased'):

    config = AutoConfig.from_pretrained(model_architecture, num_labels=len(schema), attention_mask = 1)
    model = TFAutoModelForTokenClassification.from_pretrained(model_architecture, 
                                                            config=config)
    if use_pretrained_model:
        model.bert = TFAutoModel.from_pretrained(model_path).bert
    
    lr_scheduler = LearningRateScheduler(decay_schedule)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    if model_architecture.startswith('bert'):
        model.bert._trainable = False
    elif model_architecture.startswith('distilbert'):
        model.distilbert._trainable = False

    history = model.fit(tf.constant(X_train), tf.constant(Y_train),
                    epochs=n_epochs, 
                    callbacks = [es_callback, lr_scheduler],
                    batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate*.1)
    if model_architecture.startswith('bert'):
        model.bert._trainable = True
    elif model_architecture.startswith('distilbert'):
        model.distilbert._trainable = True

    history = model.fit(tf.constant(X_train), tf.constant(Y_train),
                    epochs=n_epochs, 
                    callbacks = [es_callback, lr_scheduler],
                    batch_size=batch_size)

    return model
        
def decay_schedule(epoch, lr):
    # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
    if (epoch % 5 == 0) and (epoch != 0):
        lr = lr * 0.5
    return lr

def modelInferAndFormat(df, itemname_col, bertPreproc, schema, model, tag_lookup, lc_thresh= 0):
    
    df['input'] = df[itemname_col]
    results = modelInference(bertPreproc,schema, model,df)
    results[itemname_col] = results['input']
    results.loc[results['confidence']<lc_thresh, 'prediction'] = 'LC'


    # results['mean_item_confidence'] = results.groupby(by = [itemname_col])['confidence'].transform(lambda x: np.mean(x))
    results['min_item_confidence'] = results.groupby(by = [itemname_col])['confidence'].transform(lambda x: np.min(x))
    # results['word_position'] = results.groupby(by = [itemname_col])['word'].transform(lambda x: np.arange(len(x)))
    results['rounded_confidence'] = round(results['min_item_confidence']*10)/10
    results['word-tag'] = results[['word','prediction']].apply(lambda x: '||'.join(x),axis = 1)


    for col,val in tag_lookup.items():
        results[col] = results.groupby(by = [itemname_col])['word-tag'].transform(lambda x: tag_extract(x,val))

    out_cols = [itemname_col] + list(tag_lookup.keys()) + ['rounded_confidence']
    out = results[out_cols].drop_duplicates(subset = itemname_col).sort_values(by = 'rounded_confidence')
    
    return out

def modelInference(bertPreproc,schema, model,df):
    out = []
    for item in df['input'].values:
        out.append(
            [(w,'O') for w in item.split()]
        )
    X_test = bertPreproc.preprocess_inputs_and_labels(out,schema)
    y_probs = model.predict(X_test)[0]
    predictions = [bertPreproc.aggregate(schema, sample, predictions)
                for sample, predictions in zip(out, y_probs)]
    results = sum(predictions, [])
    results = pd.DataFrame(results, columns = ['input', 'word', 'tag', 'prediction', 'confidence']).drop_duplicates()
    return results

def tag_extract(x,val):
    return ' '.join([row.split('||')[0] for row in x if row.split('||')[1] == val])

class Classify():
    def __init__(self):
        pass

    def loadData(self, model_name, input_cols, tokenizer):
        file_list = glob.glob(f"classifier/{model_name}/data/*_labelled.csv")
        df = pd.concat([pd.read_csv(f) for f in file_list])
        train_df = df.loc[df['which_set']=='train'].sample(frac = 1).reset_index(drop = True)
        test_df = df.loc[df['which_set']=='test'].sample(frac = 1).reset_index(drop = True)

        # Remove prediction columns if they exist.
        for d in [train_df, test_df]:
            for c in list(d):
                if c in ['prediction', 'rounded_confidence']:
                    d.drop([c], axis=1, inplace=True)

        return train_df, test_df

    def preprocess_inputs(self, inputs,tokenizer):

        tokens = tokenizer.batch_encode_plus(inputs)
        inp_ids = np.array(tokens['input_ids'])
        att_msk = np.array(tokens['attention_mask'])

        max_len = np.max([len(inp) for inp in inp_ids])

        x_inp = np.zeros((len(inp_ids), max_len)).astype(int)
        x_att_msk = np.zeros((len(inp_ids), max_len)).astype(int)

        for i in range(len(inp_ids)):
            for j in range(len(inp_ids[i])):
                x_inp[i,j] = inp_ids[i][j]
                x_att_msk[i,j] = att_msk[i][j]

        return (x_inp,x_att_msk)

    def get_label_ids_dic(self, df,label_col):
        label_dic = {idx:lab for idx,lab in enumerate(df[label_col].unique())}
        conv_dic = {val:key for key,val in label_dic.items()}

        df['label_id'] = df[label_col].apply(lambda x: conv_dic[x])
        
        return df, label_dic, conv_dic

    def get_class_wts(self, df):
        vc = df['label_id'].value_counts()
        wts = 1/vc * vc.sum()/len(vc) 
        wts = {idx:wt for idx,wt in zip(wts.index,wts.values)}
        
        return dict(sorted(wts.items()))


    def load_model(self, label_dic, model_architecture, use_pretrained_model, model_path = None):

        config = AutoConfig.from_pretrained(model_architecture, num_labels=len(label_dic))
        model = TFAutoModelForSequenceClassification.from_pretrained(model_architecture, 
                                                                config=config) 

        if use_pretrained_model:
            if model_architecture.startswith('bert'):
                model.bert = TFAutoModelForSequenceClassification.from_pretrained(model_path).bert
            elif model_architecture.startswith('distilbert'):
                model.distilbert = TFAutoModelForSequenceClassification.from_pretrained(model_path).distilbert
        
        return model



    def decay_schedule(self, epoch, lr):
        # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
        if (epoch % 5 == 0) and (epoch != 0):
            lr = lr * 0.5
        return lr


    def train_model(self, model, x_train, y_train, class_wts, patience, learning_rate, batch_size,  max_n_epochs = 100):
        """
        Trains a given model on the provided training data.

        Parameters
        ----------
        model : 
            Model to be trained.
        x_train : numpy array
            Training data input.
        y_train : numpy array
            Training data labels.
        class_wts : dict
            Dictionary containing class weights.
        patience : int
            Number of epochs to wait before early stopping.
        learning_rate : float
            Learning rate for the optimizer.
        batch_size : int
            Number of samples per gradient update.
        max_n_epochs : int, optional
            Maximum number of epochs to train the model for, by default 100.

        Returns
        -------
        history : object
            Training history containing training loss, accuracy and other metrics.
    """
        
        lr_scheduler = LearningRateScheduler(decay_schedule)
        es_callback_loss = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta = 0.0001, patience=patience, restore_best_weights=True)
        es_callback_acc = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta = 0, patience=patience, restore_best_weights=True)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
        
        model.bert._trainable = False
        history = model.fit(tf.constant(x_train), tf.constant(y_train),
                        epochs=max_n_epochs, 
                        callbacks = [es_callback_loss,lr_scheduler],
                        batch_size=batch_size,
                        class_weight = class_wts,
                        verbose = 1
                        )
        
        model.bert._trainable = True
        
        #es_callback = tf.keras.callbacks.EarlyStopping(monitor=['loss','accuracy'], min_delta = 0, patience=1, restore_best_weights=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate*.1)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
        
        history = model.fit(tf.constant(x_train), tf.constant(y_train),
                        epochs=max_n_epochs, 
                        callbacks = [es_callback_loss,lr_scheduler],
                        batch_size=batch_size,
                        class_weight = class_wts,
                        verbose = 1
                        )
        
        return history

    def model_predictions(self, test_df,tokenizer,model_params,model,label_dic):
        
        inputs = self.preprocess_inputs(test_df[model_params['input_cols']].fillna('').apply(lambda x: ' [SEP] '.join(x),axis = 1).tolist(), tokenizer)[0]
        predictions = model.predict(inputs)[0]
        predictions  = softmax(predictions,axis = 1)

        test_df[f'prediction'] = [label_dic[p] for p in predictions.argmax(axis = 1)]
        test_df['rounded_confidence'] = np.round(predictions.max(axis = 1) * 10)/10
        print(test_df['rounded_confidence'].value_counts())
        
        return test_df