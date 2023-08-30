from src.arxiv_scraper import download_papers
from src.text_processing import preprocess_papers
from src.t5_training import train_t5_model
from src.t5_inference import load_trained_model, generate_title

import datetime

def main(train=False):
    topics = open("data/topics.txt", "r").read().splitlines()
    papers = download_papers(topics)
    papers = preprocess_papers(papers)

    if train:
        bleu_score = train_t5_model(papers)
        print(f"BLEU score: {bleu_score}")

    model, tokenizer = load_trained_model("./results/model")

    abstract = open("query.txt", "r").read()
    title = generate_title(model, tokenizer, abstract)
    print(f"Generated Title: {title}")

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    with open(f"result/abstract_{now}.txt", "w") as f:
        f.write(abstract)

if __name__ == "__main__":
    main(train=False)

