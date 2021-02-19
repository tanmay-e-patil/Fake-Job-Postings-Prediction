from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Get the split for CV
def get_train_test_split(c,n):
    train = []
    test = []

    for i,idx in enumerate(c):
        if idx == n:
            test.append(i)
        else:
            train.append(i)
    return train,test

# Takes the csv data and processes it into vectors 
def process_data(df):
    text_df = df[["title", "company_profile", "description", "requirements", "benefits",'employment_type',"required_experience","industry","function"]]
    text_data = combine_columns(text_df)
    labels = df["fraudulent"]
    data = tokenize_text(text_data)
    return data,labels

# Combines columns of the table
def combine_columns(df):
    text = df.apply(lambda x: ','.join(x.astype(str)),axis=1)
    return text


# Use the keras tokenizer to tokenize the text
def tokenize_text(text_data):
    num_words = 50000
    max_length = 64
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text_data)
    tokenized_data = tokenizer.texts_to_sequences(text_data)
    tokenized_data = pad_sequences(tokenized_data, maxlen=max_length)
    return tokenized_data