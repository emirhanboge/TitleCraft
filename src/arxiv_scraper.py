import requests
import feedparser
import pandas as pd

def query_arxiv(search_queryies, start=0, max_results=10):
    feeds = []
    base_url = 'http://export.arxiv.org/api/query?'
    for search_query in search_queryies:
        query = f"search_query=all:{search_query}&start={start}&max_results={max_results}"
        url = base_url + query
        response = requests.get(url)
        if response.status_code == 200:
            feed = feedparser.parse(response.content)
            feeds.extend(feed.entries)
        else:
            raise Exception(f"Failed to fetch data from arXiv with status code {response.status_code}")
    return feeds

def download_papers(topics, max_results=10):
    feed = query_arxiv(topics, max_results=max_results)
    titles = [entry.title for entry in feed]
    summaries = [entry.summary for entry in feed]
    papers = pd.DataFrame({'title': titles, 'summary': summaries})
    return papers
