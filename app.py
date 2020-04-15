from flask import Flask, render_template, url_for, request
import seq2seq_wrapper
import importlib
importlib.reload(seq2seq_wrapper)
import data_preprocessing
import data_utils_1
import data_utils_2

metadata, idx_q, idx_a = data_preprocessing.load_data(PATH = './')

# Splitting the dataset into the Training set and the Test set
(trainX, trainY), (testX, testY), (validX, validY) = data_utils_1.split_dataset(idx_q, idx_a)

# Embedding
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
vocab_twit = metadata['idx2w']
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024
idx2w, w2idx, limit = data_utils_2.get_metadata()



########## PART 2 - BUILDING THE SEQ2SEQ MODEL ##########



# Building the seq2seq model
model = seq2seq_wrapper.Seq2Seq(xseq_len = xseq_len,
                                yseq_len = yseq_len,
                                xvocab_size = xvocab_size,
                                yvocab_size = yvocab_size,
                                ckpt_path = './weights',
                                emb_dim = emb_dim,
                                num_layers = 3)



########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########



# See the Training in seq2seq_wrapper.py



########## PART 4 - TESTING THE SEQ2SEQ MODEL ##########



# Loading the weights and Running the session
session = model.restore_last_session()

# Getting the ChatBot predicted answer
def respond(question):
    encoded_question = data_utils_2.encode(question, w2idx, limit['maxq'])
    answer = model.predict(session, encoded_question)[0]
    return data_utils_2.decode(answer, idx2w)

question = []
answer = []

app = Flask(__name__)

@app.route('/')
def front_end():
    return render_template('front_end.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot', methods=['Post'])
def getvalue():
    input = request.form['Input']


    # Setting up the chat
    question.append(input)
    x = respond(input)
    answer.append(x)
    print(x)
    zip_list = zip(question,answer)
    return render_template('chatbot.html',n = zip_list)

@app.route('/Sources')
def Sources():
    return render_template('Sources.html')

if __name__ == "__main__":
    app.run(debug=True)
