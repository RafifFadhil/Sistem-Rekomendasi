import streamlit as st 
import pandas as pd 

#st.title("Sistem Rekomendasi Berita")

article_df = pd.read_excel("article.xlsx")
#st.dataframe(article_df)

#st.dataframe(novell_df.isnull().sum())
article_df = article_df[article_df['content'].notnull()]

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import re
import random

clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
sastrawi = StopWordRemoverFactory()
stopworda = sastrawi.get_stop_words()
factory=StemmerFactory()
Stemmer = factory.create_stemmer()

def clean_text(text):
  text= text.lower()
  text = clean_spcl.sub(' ', text)
  text = clean_symbol.sub(' ', text)
  text = ' '.join(word for word in text.split() if word not in stopworda)
  return text

article_df['desc_clean'] = article_df['content'].apply(clean_text)


#st.dataframe(novell_df)

#print(novell_df.columns)
article_df.set_index('content', inplace=True)
tf = TfidfVectorizer(analyzer = 'word', ngram_range=(1, 3), min_df=0.0)
tfidf_matrix = tf.fit_transform(article_df['desc_clean'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#cos_sim
#indices = novell_df.index
indices = pd.Series(article_df.index)
#indices[:15]

def recommendations (content, top = 10):

   recommended_article = []

   matching_indices = indices[indices.str.contains(content, case=False, na=False)]
   idx = matching_indices.index[0]
   score_series = pd.Series (cos_sim[idx]).sort_values (ascending = False)

   top = top + 1
   top_indexes = list (score_series.iloc[0:top].index)

   for i in top_indexes:
       recommended_article.append(list (article_df.index) [i]+" - "+str(score_series[i]))

   return recommended_article

st.title("Sistem Rekomendasi Berita")
article = st.text_input("Masukkan Judul Berita")
rekomendasi = st.button("Rekomendasi")

if rekomendasi:
    st.dataframe(recommendations(article, 15))
