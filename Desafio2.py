import pandas as pd
import spacy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
import re

# Leer el archivo CSV
df = pd.read_csv('Material_sesion5/dataset_agosto2024.csv', thousands=',')
df = df.sample(n=1000, random_state=42)  

nlp_loaded = spacy.load("model_1000_CNNv2")
nlp = spacy.load("es_core_news_lg")  

geolocator = Nominatim(user_agent="Sophia", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
geocode_cache = {}

def extract_event(text):
    sentences = text.split(".")
    return sentences[0] if len(sentences) > 0 else text

output_data = []

for index, row in df.iterrows():
    text = row["text"]
    title = row["title"]
    event = extract_event(text)
    id_news = row["id"] if 'id' in row else index

    ### Extraer categoria del modelo
    category = ""
    doc = nlp_loaded(text)
    category_score = 0
    for label, score in doc.cats.items():
        if category_score < score:
            category_score = score
            category = label
    
    ### Extraer entidades de ubicacion del texto 
    locations = []
    doc_nlp_text = nlp(text)
    doc_nlp_title = nlp(title)
    for ent in doc_nlp_text.ents:
        if ent.label_ in ["LOC", "GPE", "FAC"]:
            locations.append(ent.text)
    for ent in doc_nlp_title.ents:
        if ent.label_ in ["LOC", "GPE", "FAC"]:
            locations.append(ent.text)
            
    locations = list(set(locations))
    address = ""
    latitude = None
    longitude = None
    for location in locations:
        if location in geocode_cache:
            location_data = geocode_cache[location]
        else:
            try:
                location_data = geocode(location + ", Chile") 
                geocode_cache[location] = location_data
            except Exception as e:
                print(f"Error geocoding location {location}: {e}")
                continue
        if location_data:
            address = location_data.address
            latitude = location_data.latitude
            longitude = location_data.longitude
            break 
        
    output_data.append({
        'id_news': id_news,
        'event': event,
        'category': category,
        'address': address,
        'latitud': latitude,
        'longitud': longitude
    })
df_output = pd.DataFrame(output_data)
df_output.to_csv('output_data.csv', index=False)
