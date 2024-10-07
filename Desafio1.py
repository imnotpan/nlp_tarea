import pandas as pd
import random
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

def clean_text(text):
    text = re.sub(r'\d+\.\d+|\d{2,}', '', text)  
    text = re.sub(r'http\S+|www\S+', '', text)  
    text = re.sub(r'[,.;@#?!&$]+\ *', ' ', text)  
    return text

train_data = pd.read_csv('Material_sesion5/train_data.csv', thousands=',')
train_data['date'] = pd.to_datetime(train_data['date'], format='%b %d, %Y @ %H:%M:%S.%f')
train_data['date'] = train_data['date'].dt.strftime('%Y-%m-%d')

train_data['clean_text'] = train_data['text'].apply(clean_text)
random_rows = train_data.sample(n=7500, random_state=42)
unique_classes = random_rows["clase"].unique()

# print(train_data)
# print(unique_classes)

data_list = []
for _, row in random_rows.iterrows():
    text = row["clean_text"]  
    clase = row["clase"]
    cats = {cls: 1 if cls == clase else 0 for cls in unique_classes}
    data_list.append((text, {"cats": cats}))

TRAIN_DATA = data_list
nlp = spacy.load("es_core_news_sm")

train_examples = []

for example in TRAIN_DATA:
    train_examples.append(Example.from_dict(nlp.make_doc(example[0]), example[1]))

def get_examples():
    return train_examples

model = {
            "@architectures": "spacy.TextCatCNN.v2",
            "exclusive_classes": True,
            "tok2vec": DEFAULT_TOK2VEC_MODEL,
        }

textcat = nlp.add_pipe("textcat", config={"model": model})
textcat.initialize(get_examples)
with nlp.select_pipes(enable="textcat"):
    optimizer = nlp.begin_training()
    for epoch in range(100):
        losses = {}
        random.shuffle(TRAIN_DATA)
        # Dividir los datos en lotes y actualizar el modelo
        for batch in minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001)):
            texts, annotations = zip(*batch)
            example = []
            # Actualizar el modelo con iteraciones
            for i in range(len(texts)):
                doc = nlp.make_doc(texts[i])
                example.append(Example.from_dict(doc, annotations[i]))
            nlp.update(example, drop=0.5, losses=losses)
        print(losses)
        
nlp.to_disk("model_1000_CNNv2")
