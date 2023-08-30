# TitleCraft: Academic Title Generator

This project aims to download papers from Arxiv based on specified topics, preprocess the paper summaries, and train a T5 model to generate titles for the papers. The project also provides functionalities to load a pre-trained model and generate titles for new summaries.

## Installation

Clone the repository:

```bash
git clone https://github.com/emirhanboge/TitleCraft.git
```

Navigate into the project directory:
```bash
cd TitleCraft
```

Install required Python packages:
```bash
pip install -r requirements.txt
```

TitleCraft/
|-- src/
|   |-- arxiv_scraper.py
|   |-- text_processing.py
|   |-- t5_training.py
|   |--inference.py
|-- main.py
|-- requirements.txt
|-- README.md

## Usage
```bash
python3 main.py
```

The above command will download papers based on predefined topics (data/topics.txt), preprocess them, train a T5 model (if specified), and save the model. After the training, it will load the model and generate a title for an sample abstract (query.txt).
