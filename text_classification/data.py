"""Download the attributed UCI corpus locally and fit text preprocessing on train only."""

import argparse
from collections import Counter
import hashlib
from io import BytesIO
from pathlib import Path
import re
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "sms-spam"
DATA_FILE = DATA_DIRECTORY / "SMSSpamCollection"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SOURCE_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
LABELS = {"ham": 0, "spam": 1}
PAD, UNK = "<pad>", "<unk>"


def download_data(output=DATA_DIRECTORY):
    """Fetch only the official data and its readme; never extract arbitrary ZIP paths."""
    output = Path(output).resolve()
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repository) and not any(
            output.is_relative_to(repository / name) for name in ("data", "datasets", "work")):
        raise ValueError("Inside the repository, keep the corpus under data/, datasets/, or work/.")
    path = output / "SMSSpamCollection"
    if path.exists():
        read_messages(path)  # Check the existing file before reusing it.
        return path
    with urlopen(SOURCE_URL, timeout=60) as response:
        downloaded = response.read()
    with ZipFile(BytesIO(downloaded)) as archive:
        corpus, readme = archive.read("SMSSpamCollection"), archive.read("readme")
    if hashlib.sha256(corpus).hexdigest() != SOURCE_SHA256:
        raise ValueError("The official download has changed. Review it before updating the checksum.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "original-readme.txt").write_bytes(readme)
    path.write_bytes(corpus)
    return path


def read_messages(path=DATA_FILE, expected_sha256=SOURCE_SHA256):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Run python -m text_classification.data --download first.")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Unexpected corpus checksum; use the documented official UCI version.")
    # Split only on the first tab; quoted text is content, not a CSV escape rule.
    rows = [line.split("\t", 1) for line in content.decode("utf-8").splitlines()]
    if any(len(row) != 2 or row[0] not in LABELS for row in rows):
        raise ValueError("Expected one ham/spam label and tab-separated message per line.")
    return pd.DataFrame(rows, columns=["label", "text"])


def tokenize(text):
    """Lowercase and extract Unicode word/number tokens; punctuation separates words."""
    return re.findall(r"\b\w+\b", text.lower())


def prepare_messages(frame, seed=42):
    """Remove normalized duplicates, then stratify 70/15/15 without using features."""
    if (not {"label", "text"}.issubset(frame.columns) or frame.empty
            or not frame["label"].isin(LABELS).all()
            or not frame["text"].map(lambda value: isinstance(value, str)).all()):
        raise ValueError("Expected nonempty text records labeled ham or spam.")
    frame = frame[["label", "text"]].copy()
    frame["normalized"] = frame["text"].map(lambda text: " ".join(tokenize(text)))
    if (frame.groupby("normalized")["label"].nunique() > 1).any():
        raise ValueError("Identical normalized messages have conflicting labels; review the data.")
    unique = frame.drop_duplicates("normalized").reset_index(drop=True)
    labels = unique["label"].map(LABELS).to_numpy(dtype=np.int64)
    if unique["label"].nunique() != 2 or unique["label"].value_counts().min() < 10:
        raise ValueError("Use at least ten distinct messages per class for three stratified splits.")
    indices = np.arange(len(unique))
    development, test = train_test_split(indices, test_size=0.15, stratify=labels, random_state=seed)
    train, validation = train_test_split(development, test_size=0.15 / 0.85,
                                         stratify=labels[development], random_state=seed)
    return {"texts": unique["normalized"].tolist(), "labels": labels,
            "train": train, "validation": validation, "test": test,
            "source_rows": len(frame), "source_class_counts": frame["label"].value_counts().to_dict(),
            "duplicates_removed": len(frame) - len(unique)}


def build_vocabulary(texts, max_words=10000):
    """Call with training messages only; 0 is padding and 1 is unknown."""
    if max_words < 1:
        raise ValueError("max_words must be positive.")
    counts = Counter(word for text in texts for word in tokenize(text))
    words = sorted(counts, key=lambda word: (-counts[word], word))[:max_words]
    return {PAD: 0, UNK: 1, **{word: index + 2 for index, word in enumerate(words)}}


def encode_messages(texts, vocabulary, max_length=60):
    """Keep first max_length tokens, right-pad, and represent empty text by UNK."""
    if max_length < 1 or vocabulary.get(PAD) != 0 or vocabulary.get(UNK) != 1:
        raise ValueError("Use a positive maximum length and reserved PAD=0 / UNK=1 IDs.")
    encoded = torch.zeros((len(texts), max_length), dtype=torch.long)
    lengths = torch.empty(len(texts), dtype=torch.long)
    for index, text in enumerate(texts):
        ids = [vocabulary.get(token, 1) for token in tokenize(text)[:max_length]] or [1]
        lengths[index] = len(ids)
        encoded[index, :len(ids)] = torch.tensor(ids)
    return encoded, lengths


def load_data(path=DATA_FILE, seed=42, max_length=60, max_words=10000):
    data = prepare_messages(read_messages(path), seed)
    vocabulary = build_vocabulary([data["texts"][i] for i in data["train"]], max_words)
    tokens, lengths = encode_messages(data["texts"], vocabulary, max_length)
    data.update(tokens=tokens, lengths=lengths, vocabulary=vocabulary,
                data_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the official corpus to a Git-ignored folder.")
    args = parser.parse_args()
    path = download_data() if args.download else DATA_FILE
    frame = read_messages(path)
    print(f"Local corpus: {len(frame)} messages; {int(frame.label.eq('spam').sum())} spam.")
    print("Source: UCI SMS Spam Collection, CC BY 4.0. Raw messages are not exported.")


if __name__ == "__main__":
    main()
