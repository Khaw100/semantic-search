import torch
import numpy as np
import random
import re
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time
import gensim
from gensim.models.fasttext import load_facebook_model

# Preprocessor Class
class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def remove_punctuation(self, text):
        punct_pattern = re.compile(r'[^\w\s]')
        return punct_pattern.sub(' ', text)

    def remove_stopwords(self, text):
        filtered_text = [word for word in text.split() if word.lower() not in self.stop_words]
        return ' '.join(filtered_text)

    def remove_extra_whitespaces(self, text):
        whitespace_pattern = re.compile(r'\s+')
        return whitespace_pattern.sub(' ', text)

    def remove_numbers(self, text):
        number_pattern = re.compile(r'\d+')
        return number_pattern.sub(' ', text)

    def stem_text_porter(self, text):
        stemmed_words = [self.stemmer.stem(word) for word in text.split()]
        return ' '.join(stemmed_words)
    
    def preprocessing(self, text):
        text = text.lower()
        text = self.remove_punctuation(text)
        text = self.remove_extra_whitespaces(text)
        return text

    def preprocessing_with_stemming(self, text):
        text = text.lower()
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_stopwords(text)
        text = self.remove_extra_whitespaces(text)
        text = self.stem_text_porter(text)
        return text

    def preprocessing_without_stemming(self, text):
        text = text.lower()
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        text = self.remove_extra_whitespaces(text)
        return text

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set the seed
SEED = 42
set_seed(SEED)

# SBERT Class
class SBERTModel:
    def __init__(self, data, pretrained_model):
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(pretrained_model, device=self.device)
        self.preprocessor = Preprocessor()
        print(self.model)

    def fit_transform_stemmed(self):
        self.course_embeddings = self.model.encode(self.data['preprocessed_description_stemmed'].tolist()) 

    def fit_transform_unstemmed(self):
        self.course_embeddings = self.model.encode(self.data['preprocessed_description_unstemmed'].tolist())

    def semantic_search_unstemmed(self, query, top_n=5, similarity_threshold=0.07):
        start_time = time.time()
        preprocessed_query = self.preprocessor.remove_stopwords(self.preprocessor.remove_punctuation(query))
        query_embedding = self.model.encode(preprocessed_query, convert_to_tensor=True)

        course_tensor = torch.tensor(self.course_embeddings, dtype=torch.float32).to(self.device)

        cos_similarities = torch.nn.functional.cosine_similarity(course_tensor, query_embedding, dim=1)
        top_indices = torch.argsort(cos_similarities, descending=True)

        results = {"results":[]}
        
        indices_above_threshold = top_indices[cos_similarities[top_indices] > similarity_threshold]
        
        for index in indices_above_threshold:
            filtered_item = {
                'item_id' : self.data.at[index.item(),'course_id'],
                'item_title': self.data.at[index.item(), 'title'],
                'item_translated_title': self.data.at[index.item(), 'translated_title'],
                'item_headline': self.data.at[index.item(), 'headline'],
                'item_translated_headline': self.data.at[index.item(), 'translated_headline'],
                'item_objectives': self.data.at[index.item(), 'objectives_summary'],
                'item_translated_objectives': self.data.at[index.item(), 'translated_objectives_summary'],
                'item_level': self.data.at[index.item(), 'instructional_level'],
                'item_score': cos_similarities[index].item()
            }
            results["results"].append(filtered_item)
        end_time = time.time()
        search_time = end_time - start_time
        return results, search_time

    def semantic_search_stemmed(self, query, top_n=5, similarity_threshold=0.07):
        start_time = time.time()
        preprocessed_query = self.preprocessor.remove_stopwords(self.preprocessor.remove_punctuation(query))
        query_embedding = self.model.encode(preprocessed_query, convert_to_tensor=True)

        course_tensor = torch.tensor(self.course_embeddings, dtype=torch.float32).to(self.device)

        cos_similarities = torch.nn.functional.cosine_similarity(course_tensor, query_embedding, dim=1)
        top_indices = torch.argsort(cos_similarities, descending=True)

        results = {"results":[]}
        
        indices_above_threshold = top_indices[cos_similarities[top_indices] > similarity_threshold]
        
        for index in indices_above_threshold:
            filtered_item = {
                'item_id' : self.data.at[index.item(),'course_id'],
                'item_title': self.data.at[index.item(), 'title'],
                'item_translated_title': self.data.at[index.item(), 'translated_title'],
                'item_headline': self.data.at[index.item(), 'headline'],
                'item_translated_headline': self.data.at[index.item(), 'translated_headline'],
                'item_objectives': self.data.at[index.item(), 'objectives_summary'],
                'item_translated_objectives': self.data.at[index.item(), 'translated_objectives_summary'],
                'item_level': self.data.at[index.item(), 'instructional_level'],
                'item_score': cos_similarities[index].item()
                
            }
            results["results"].append(filtered_item)
        end_time = time.time()
        search_time = end_time - start_time
        return results, search_time

# FastText + SBERTClass
class FastTextSBERTModel:
    def __init__(self, data, pretrained_model):
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(pretrained_model, device=self.device)
        self.model_ft = load_facebook_model("cc.en.300.bin")
        self.preprocessor = Preprocessor()

    def fit_transform_stemmed_ft(self):
        self.course_embeddings = self.model.encode(self.data['preprocessed_description_stemmed'].tolist())

    def fit_transform_unstemmed_ft(self):
        self.course_embeddings = self.model.encode(self.data['preprocessed_description_unstemmed'].tolist())

    def semantic_search_stemmed_ft(self, query, top_n=5, similarity_threshold=0.07, n_similar=30):
        start_time = time.time()
        preprocessed_query = self.preprocessor.preprocessing_with_stemming(query)
        expanded_query = []
        ex_query = []
        temp_query = self.preprocessor.remove_stopwords(self.preprocessor.remove_punctuation(query))
        for word in temp_query.split():
            similar_words = [word] + [w for w, sim in self.model_ft.wv.most_similar(word, topn=n_similar)]
            expanded_query.extend(similar_words)
        preprocessed_expanded_query = self.preprocessor.preprocessing(' '.join(expanded_query))
        words = preprocessed_expanded_query.split()
        seen_map = {}
        for item in words:
            if item not in seen_map:
                ex_query.append(item)
                seen_map[item] = True
        final_query = " ".join(ex_query)
        query_embedding = self.model.encode(final_query, convert_to_tensor=True)

        course_tensor = torch.tensor(self.course_embeddings, dtype=torch.float32).to(self.device)

        cos_similarities = torch.nn.functional.cosine_similarity(course_tensor, query_embedding, dim=1)
        top_indices = torch.argsort(cos_similarities, descending=True)

        results = {"results": []}
        for index in top_indices:
            if cos_similarities[index] > similarity_threshold:
                filtered_item = {
                    'item_id': self.data.at[index.item(), 'course_id'],
                    'item_title': self.data.at[index.item(), 'title'],
                    'item_translated_title': self.data.at[index.item(), 'translated_title'],
                    'item_headline': self.data.at[index.item(), 'headline'],
                    'item_translated_headline': self.data.at[index.item(), 'translated_headline'],
                    'item_objectives': self.data.at[index.item(), 'objectives_summary'],
                    'item_translated_objectives': self.data.at[index.item(), 'translated_objectives_summary'],
                    'item_level': self.data.at[index.item(), 'instructional_level'],
                    'item_score': cos_similarities[index].item()
                }
                results["results"].append(filtered_item)
        end_time = time.time()
        search_time = end_time - start_time
        return results, search_time

    def semantic_search_unstemmed_ft(self, query, top_n=5, similarity_threshold=0.07, n_similar=30):
        start_time = time.time()
        expanded_query = []
        ex_query = []
        temp_query = self.preprocessor.remove_stopwords(self.preprocessor.remove_punctuation(query))
        for word in temp_query.split():
            similar_words = [word] + [w for w, sim in self.model_ft.wv.most_similar(word, topn=n_similar)]
            expanded_query.extend(similar_words)
        preprocessed_expanded_query = self.preprocessor.preprocessing(' '.join(expanded_query))
        words = preprocessed_expanded_query.split()
        seen_map = {}
        for item in words:
            if item not in seen_map:
                ex_query.append(item)
                seen_map[item] = True
                
        final_query = " ".join(ex_query)
        query_embedding = self.model.encode(final_query, convert_to_tensor=True)

        course_tensor = torch.tensor(self.course_embeddings, dtype=torch.float32).to(self.device)

        cos_similarities = torch.nn.functional.cosine_similarity(course_tensor, query_embedding, dim=1)
        top_indices = torch.argsort(cos_similarities, descending=True)

        results = {"results": []}
        for index in top_indices:
            if cos_similarities[index] > similarity_threshold:
                filtered_item = {
                    'item_id': self.data.at[index.item(), 'course_id'],
                    'item_title': self.data.at[index.item(), 'title'],
                    'item_translated_title': self.data.at[index.item(), 'translated_title'],
                    'item_headline': self.data.at[index.item(), 'headline'],
                    'item_translated_headline': self.data.at[index.item(), 'translated_headline'],
                    'item_objectives': self.data.at[index.item(), 'objectives_summary'],
                    'item_translated_objectives': self.data.at[index.item(), 'translated_objectives_summary'],
                    'item_level': self.data.at[index.item(), 'instructional_level'],
                    'item_score': cos_similarities[index].item()
                }
                results["results"].append(filtered_item)
        end_time = time.time()
        search_time = end_time - start_time
        return results, search_time
