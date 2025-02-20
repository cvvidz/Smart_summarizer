#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Import necessary libraries
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from gtts import gTTS
import os
import numpy as np
import networkx as nx
from playsound import playsound

# Function to read the article
def read_article(file_name):
    with open(file_name, "r") as file:
        filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = [sentence.replace("[^a-zA-Z]", " ").split(" ") for sentence in article]
    if sentences:
        sentences.pop()  # Remove the last empty item if exists
    return sentences

# Function to calculate sentence similarity
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w not in stopwords:
            vector1[all_words.index(w)] += 1
    for w in sent2:
        if w not in stopwords:
            vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

# Function to build similarity matrix
def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix

# Function to generate summary
def generate_summary(file_name, top_n):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read the text and split into sentences
    sentences = read_article(file_name)

    # Step 2 - Generate Similarity Matrix
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort sentences by rank
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Step 5 - Select top sentences for summary
    for i in range(min(top_n, len(ranked_sentence))):  # Ensure it doesn't exceed available sentences
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 6 - Convert summary to speech
    summary_text = " ".join(summarize_text)
    print("\nGenerated Summary:\n", summary_text)
    
    output = gTTS(text=summary_text, lang='en', slow=False)
    output.save("summary.mp3")
    playsound("summary.mp3")

#Input

file_name = input("Enter the text file name (with extension, e.g., sample.txt): ")
top_n = int(input("Enter the number of sentences for the summary: "))

generate_summary(file_name, top_n)

