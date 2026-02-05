# BPE Tokenizer — Test Artifacts

After running the BPE training and tokenizer tests from the **project root**:

```bash
uv run pytest cs336_basics/BPE_Tokenizer/tests
```

you should observe the following folder structure inside `cs336_basics/BPE_Tokenizer/tests/`:

```bash
tests/
├── README.md # This file
├── test_01_train_bpe.py  # The test script of BPE trainer
├── test_01_train_bpe.txt # The test log of BPE trainer
├── test_02_tokenizer.py  # The test script of BPE tokenizer
├── test_02_tokenizer.txt # The test log of BPE tokenizer
├── TinyStoriesV2-GPT4-train-vocab.pkl  # Vocabulary trained on TinyStories (pickle)
├── TinyStoriesV2-GPT4-train-vocab.json # Vocabulary trained on TinyStories (JSON for inspection)
├── TinyStoriesV2-GPT4-train-merges.pkl # Merges trained on TinyStories (pickle)
├── TinyStoriesV2-GPT4-train-merges.txt # Merges trained on TinyStories (TXT for inspection)
├── TinyStoriesV2-GPT4-train.npy # Token IDs of TinyStories training set
├── TinyStoriesV2-GPT4-valid.npy # Token IDs of TinyStories validation set
├── owt_train-vocab.pkl  # Vocabulary trained on OpenWebText (pickle)
├── owt_train-vocab.json # Vocabulary trained on OpenWebText (JSON for inspection)
├── owt_train-merges.pkl # Merges trained on OpenWebText (pickle)
├── owt_train-merges.txt # Merges trained on OpenWebText (TXT for inspection)
├── owt_train.npy # Token IDs of OpenWebText training set
└── owt_valid.npy # Token IDs of OpenWebText validation set
```

Note that this process takes up to two hours on a Linux server with 128 cores.