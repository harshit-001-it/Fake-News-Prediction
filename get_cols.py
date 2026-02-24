import pandas as pd
import os

paths = {
    "fake_news_1": r"C:\Users\harsh_2pgm3oe\OneDrive\Documents\Coding\All Docs\Fake News\fake news 1\WELFake_Dataset.csv",
    "fake_news_2_fake": r"C:\Users\harsh_2pgm3oe\OneDrive\Documents\Coding\All Docs\Fake News\fake news 2\fake.csv",
    "fake_news_3": r"C:\Users\harsh_2pgm3oe\OneDrive\Documents\Coding\All Docs\Fake News\fake news 3\FakeNewsNet.csv",
    "fake_news_4_fake": r"C:\Users\harsh_2pgm3oe\OneDrive\Documents\Coding\All Docs\Fake News\fake news 4\Fake.csv",
    "fake_news_4_true": r"C:\Users\harsh_2pgm3oe\OneDrive\Documents\Coding\All Docs\Fake News\fake news 4\True.csv"
}

with open('cols.txt', 'w') as f:
    for name, path in paths.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, nrows=0)
                f.write(f"{name} columns: {list(df.columns)}\n")
            except Exception as e:
                f.write(f"{name} error: {e}\n")
        else:
            f.write(f"{name} not found\n")
