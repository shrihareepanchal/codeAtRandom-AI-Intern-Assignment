from sklearn.datasets import fetch_20newsgroups
import os

# Directory inside your project
docs_dir = "data/docs"
os.makedirs(docs_dir, exist_ok=True)

print("Downloading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Limit to 200 files for faster indexing
max_files = 200
print(f"Saving first {max_files} documents to .txt files...")

for i, text in enumerate(newsgroups.data[:max_files]):
    file_path = os.path.join(docs_dir, f"doc_{i+1}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

print(f"Successfully saved {max_files} text files in '{docs_dir}'")
