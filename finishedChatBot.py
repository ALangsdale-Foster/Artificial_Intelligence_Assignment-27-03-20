from __future__ import absolute_import, division, print_function, unicode_literals
import random
import aiml
import nltk
import numpy as np
import sklearn
import string
##import wikipediaapi
import json, requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
print(tf.__version__)
import cv2
import glob
import pandas as pd
import os, re
import tensorflow_datasets as tfds
import gym

#Create env/agent
env = gym.make("CartPole-v1")
alpha = 0.5
gamma = 0.90
epsilon = 0.1
batch_size = 32

def model(inputs, layer_size, ):
    for i, size in enumerate(layer_size):
        inputs = tf.layers.Dense(inputs, size, activation = "relu" if i < len(layer_size) - 1 else None)
    return inputs

class QAgent:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def get_Q(self, state, action):
        #print("Q: " + str(self.q.get((state, action), 0.0)))
        return self.q.get((state, action), 0.0)

    def learn_Q(self, state, action, reward, value):
        old_val = self.q.get((state,action), None)

        if old_val is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = old_val + self.alpha * (value - old_val)

    def act(self, state, return_q = False):
        q = [self.get_Q(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
        count = q.count(maxQ)

        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]

        if return_q:
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        new_maxQ = max([self.get_Q(state2, a) for a in self.actions])
        self.learn_Q(state1, action1, reward, reward + self.gamma * new_maxQ)
        

#Descretize space
def build_state(features):    
    return int("".join(map(lambda feature: str(int(feature)), features)))

def bin(value, bins):
    return np.digitize(x=[value], bins = bins)[0]

movie_lines_path = "movie_lines.txt"
movie_conversations_path = "movie_conversations.txt"

#Preprocess/Load Data

MAX_SAMPLES = 50000

def preprocess_sentence(sentence):
  #removes trailing/leading spaces in lower case form
  sentence = sentence.lower().strip()
  
  #add space between word/punctuation
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)

  #replaces non alphabet chars/punctuation with space
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()

  return sentence

def load_conversations():
  #dict storing line id + text
  lineID = {}
  with open(movie_lines_path, errors = "ignore") as file:
    lines = file.readlines()
  
  for line in lines:
    #removing dataset specific formatting
    parts = line.replace("\n", "").split(' +++$+++ ')
    lineID[parts[0]] = parts[4]
  
  inputs, outputs = [], []

  
  with open(movie_conversations_path, "r") as file:
    lines = file.readlines()
  
  for line in lines:
    #removing dataset specific formatting
    parts = line.replace("\n", "").split(" +++$+++ ")
    
    #fetch conversation from ID list
    conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
    for i in range(len(conversation) - 1):
      inputs.append(preprocess_sentence(lineID[conversation[i]]))
      outputs.append(preprocess_sentence(lineID[conversation[i + 1]]))

      if len(inputs) >= MAX_SAMPLES:
        return inputs, outputs
  
  return inputs, outputs

questions, answers = load_conversations()

#Tokenizer time
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size = 2**13)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

#Max sentence len
MAX_LENGTH = 40

#tokenize, filter, pad
def tokenize_and_filter(inputs, outputs):
  
  tokenized_inputs, tokenized_outputs = [], []
  #tokenize
  for(sentence1, sentence2) in zip(inputs, outputs):
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  #pad
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)

#Creation of tf dataset
BATCH_SIZE = 64
BUFFER_SIZE = 20000

#remove start token
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

#Scaled dot product attention - yay more matrix stuff :(
def scaled_dot_product_attention(query, key, value, mask):
  #calc weights
  matmul_qk = tf.matmul(query, key, transpose_b = True)

  #scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  #add mask to zero out padding
  if mask is not None:
    logits += (mask * -1e9)
  
  #normalize soft max on last axis
  attention_weights = tf.nn.softmax(logits, axis = -1)


  output = tf.matmul(attention_weights, value)

  return output

#Multi head attention

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, name = "multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name = name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads
    self.query_dense = tf.keras.layers.Dense(units = d_model)
    self.key_dense = tf.keras.layers.Dense(units = d_model)
    self.value_dense = tf.keras.layers.Dense(units = d_model)
    self.dense = tf.keras.layers.Dense(units = d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs["query"], inputs["key"], inputs["value"], inputs["mask"]
    batch_size = tf.shape(query)[0]

    #linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    #split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    #scaled dot product attnetion
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)
    scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs

#transformer masking
def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  
  return tf.maximum(look_ahead_mask, padding_mask)

#Positional encoding
class PositionalEncoding(tf.keras.layers.Layer):
  
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))

    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    #sin(even array indices)
    sines = tf.math.sin(angle_rads[:, 0::2])
    #cos(odd array indices)
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis = -1)
    pos_encoding = pos_encoding[tf.newaxis, ...]

    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

#Encoder layer
def encoder_layer(units, d_model, num_heads, dropout, name = "encoder_layer"):
  inputs = tf.keras.Input(shape = (None, d_model), name = "inputs")
  padding_mask = tf.keras.Input(shape = (1, 1, None), name = "padding_mask")

  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate = dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon = 1e-6)(inputs + attention)
  
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)
  
  return tf.keras.Model(inputs = [inputs, padding_mask], outputs = outputs, name = name)

#Encoder
def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
  inputs = tf.keras.Input(shape = (None,), name = "inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
  
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate = dropout)(embeddings)

  for i in range(num_layers):
    outputs = encoder_layer(units = units, d_model = d_model, num_heads = num_heads, dropout = dropout, 
                            name="encoder_layer_{}".format(i),)([outputs, padding_mask])
  return tf.keras.Model(inputs = [inputs, padding_mask], outputs = outputs, name = name)

#Decoder layer
def decoder_layer(units, d_model, num_heads, dropout, name = "decoder_layer"):
  inputs = tf.keras.Input(shape = (None, d_model), name = "inputs")
  enc_outputs = tf.keras.Input(shape = (None, d_model), name = "encoder_outputs")
  look_ahead_mask = tf.keras.Input(shape = (1, None, None), name = "look_ahead_mask")
  padding_mask = tf.keras.Input(shape = (1, 1, None), name = "padding_mask")

  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  attention1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(attention1 + inputs)

  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate = dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(inputs = [inputs, enc_outputs, look_ahead_mask, padding_mask], outputs = outputs, name = name)

#decoder
def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='decoder'):
  inputs = tf.keras.Input(shape = (None,), name = "inputs")
  enc_outputs = tf.keras.Input(shape = (None, d_model), name = "encoder_outputs")
  look_ahead_mask = tf.keras.Input(shape = (1, None, None), name = "look_ahead_mask")
  padding_mask = tf.keras.Input(shape = (1, 1, None), name = "padding_mask")

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs = outputs, name = name)

#Transformer
def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape = (None,), name = "inputs")
  dec_inputs = tf.keras.Input(shape = (None,), name = "dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

  #mask decoder inputs at 1st attention
  look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)
  #mask encoder outputs at 2nd attention
  dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(inputs)

  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

#Initialize model

tf.keras.backend.clear_session()

  #Params
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

model = transformer(vocab_size = VOCAB_SIZE, num_layers = NUM_LAYERS, units = UNITS, d_model = D_MODEL, num_heads = NUM_HEADS, 
                    dropout = DROPOUT)

#Loss func
def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape = (-1, MAX_LENGTH - 1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction = "none")(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)

#Custom learning rate

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps = 4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

#Model compliation
learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon = 1e-9)

def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape = (-1, MAX_LENGTH - 1))
  
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)



#model.load_weights("transformerWeights.h5") //weights broken - causes entire program to crash (some kind of discrepency between local version/google colab version - model works fine on colab but not local
model.compile(optimizer = optimizer, loss = loss_function, metrics = [accuracy])

def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence
 
##REFERENCE: https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e used for lemmatization, TFIDF approach

##wikiPage = wikipediaapi.Wikipedia('en')
##wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

##v = """
##stalingrad => {}
##leningrad => {}
##moscow => {}
##tobruk => {}
##anzio => {}
##rome => {}
##berlin => {}
##okinawa => {}
##wizna => {}
##russia => f1
##libya => f2
##italy => f3
##germany => f4
##japan => f5
##poland =>f6
##be_in => {}
##"""

v = """
stalingrad => {}
leningrad => {}
moscow => {}
tobruk => {}
anzio => {}
rome => {}
berlin => {}
okinawa => {}
russia => f1
libya => f2
italy => f3
germany => f4

be_in => {}
"""
folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'
objectCounter = 0


questionanswer = open('questionanswer.txt')
rawFile = questionanswer.read()
rawFile = rawFile.lower()

kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

nltk.download("wordnet")
nltk.download("punkt")

sent_tokens = nltk.sent_tokenize(rawFile) ##generate tokens based on file
word_tokens = nltk.word_tokenize(rawFile)

lemmer = nltk.stem.WordNetLemmatizer() ##initialize lemmatizer

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

json_file = open("newModel.json", "r") ##loading nueral network model
json_loaded_model = json_file.read()
json_file.close()

with CustomObjectScope({"GlorotUniform": glorot_uniform()}):
    loaded_model = model_from_json(json_loaded_model)
loaded_model.load_weights("newWeights.h5")

def response(userInput): ##generate response function
    robo_response=''
    sent_tokens.append(userInput)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response  

def LemTokens(tokens): ##lemmatize input
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text): ##normalize input to lower case
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



  
print("MAIN PROGRAM")
while True:
    #get user input
    try:
        userInput = input(">> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    responseAgent = 'aiml'

    if userInput[0:8] == "classify":
        responseAgent = "neural"
        
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)

    if responseAgent == "neural":

        userPath = userInput[9:]
        data = []

        img_path = userPath
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        img = np.reshape(img,[1,100,100,3])

        prediction = loaded_model.predict(img)
        
        if prediction[0][0] == 1:
            prediction = "Cargo"
        elif prediction[0][1] == 1:
            prediction = "Military"
        elif prediction[0][2] == 1:
            prediction = "Carrier",
        elif prediction[0][3] == 1:
            prediction = "Cruise"
        elif prediction[0][4] == 1:
            prediction = "Tanker"
        else:
            prediction = "Unknown"
        
        answer = "This image is a " + prediction + " ship"
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
##            wpage = wikiPage.page(params[1])
##            if wpage.exists():
##                print(wpage.summary)
##                print("More information at ", wpage.canonicalurl)
##            else:
##                print("Unknown Topic.")
                print("lol")
        elif cmd == 2:
            Atxt = response(userInput).split('*')
            print(Atxt[1])
            sent_tokens.remove(userInput)
        elif cmd == 4:
            params[1] = params[1].replace(" ", "")
            params[2] = params[2].replace(" ", "")
            
            params[1] = params[1].lower()
            params[2] = params[2].lower()

            if (params[1] in folval) and (params[2] in folval):
                o = 'o' + str(objectCounter)
                objectCounter +=1
                folval['o' + o] = o

                if len(folval[params[1]]) == 1:
                    folval[params[1]].clear()

                folval[params[1]].add((o,))

                if len(folval["be_in"]) == 1:
                    if ('',) in folval["be_in"]:
                        folval["be_in"].clear()
                folval["be_in"].add((o, folval[params[2]]))
            else:
                if params[1] not in folval:
                    print("Battle not found, please try again")
                if params[2] not in folval:
                    print("Country not found, please try again")

        elif cmd == 5:
            params[1] = params[1].replace(" ", "")
            params[2] = params[2].replace(" ", "")
            
            params[1] = params[1].lower()
            params[2] = params[2].lower()
            
            if (params[1] in folval) and (params[2] in folval):
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'all ' + params[1] + ' are_in ' + params[2]               
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]

                if results[2] == True:
                    print("Yes.")
                else:
                    print("No.")

            else:
                if params[1] not in folval:
                    print("Battle not found, please try again")
                if params[2] not in folval:
                    print("Country not found, please try again")

        elif cmd == 7:
            params[1] = params[1].replace(" ", "")
            params[1] = params[1].lower()

            if params[1] in folval:
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                e = nltk.Expression.fromstring("be_in(x," + params[1] + ")")
                sat = m.satisfiers(e, "x", g)
                if len(sat) == 0:
                    print("None.")
                else:
                    sol = folval.values()
                    for so in sat:
                        for k, v in folval.items():
                            if len(v) > 0:
                                vl = list(v)
                                if len(vl[0]) == 1:
                                    for i in vl:
                                        if i[0] == so:
                                            print(k)
                                            break
            else:
                if params[1] not in folval:
                    print("Battle not found, please try again")

        elif cmd == 8:
            env = gym.make("CartPole-v1")
            average_goal = 195
            max_steps = 200
            last_time_steps = np.ndarray(0)
            n_bins = 8
            n_bins_angle = 10

            features_num = env.observation_space.shape[0]
            last_time_steps = np.ndarray(0)

            cart_position_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
            pole_angle_bins = pd.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
            cart_velocity_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
            angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

            qLearning = QAgent(actions = range(env.action_space.n), alpha = alpha, gamma = gamma, epsilon = epsilon)

            try:
                for i_episode in range(1500):
                    observation = env.reset()

                    cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
                    state = build_state([bin(cart_position, cart_position_bins), bin(pole_angle, pole_angle_bins), bin(cart_velocity, cart_velocity_bins), bin(angle_rate_of_change, angle_rate_bins)])

                    for t in range(max_steps):
                        env.render()

                        action = qLearning.act(state)
                        observation, reward, done, info = env.step(action)

                        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

                        nextState = build_state([bin(cart_position, cart_position_bins), bin(pole_angle, pole_angle_bins), bin(cart_velocity, cart_velocity_bins), bin(angle_rate_of_change, angle_rate_bins)])

                        if not(done):
                            qLearning.learn(state, action, reward, nextState)
                            state = nextState
                        else:
                            reward = -200
                            qLearning.learn(state, action, reward, nextState)
                            last_time_steps = np.append(last_time_steps, [int(t + 1)])
                            break

                    l = last_time_steps.tolist()
                    l.sort()
                    print("Score: {:0.2f}".format(last_time_steps.mean()))
            except KeyboardInterrupt:
                print("Keyboard Interrupt Received, shutting down game")
            finally:
                env.close()
        elif cmd == 99:
            print("Unknown Input, please try again")
    else:
        print(answer)

    
