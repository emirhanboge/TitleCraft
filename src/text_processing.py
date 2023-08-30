import pandas as pd
import string
from sklearn.model_selection import train_test_split

def preprocess_papers(papers):
    papers = papers.drop_duplicates(subset=['title'])
    papers = papers.reset_index(drop=True)

    for i in range(len(papers)):
        papers['summary'][i] = papers['summary'][i].lower()
        new_summary = ''.join(char for char in papers['summary'][i] if char not in string.punctuation)
        papers['summary'][i] = new_summary

        papers['title'][i] = papers['title'][i].lower()
        new_title = ''.join(char for char in papers['title'][i] if char not in string.punctuation)
        papers['title'][i] = new_title

    return papers

