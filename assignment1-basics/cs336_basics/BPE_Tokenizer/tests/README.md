# BPE Tokenizer — Test Artifacts

After running the BPE training test from the **project root**:

```bash
uv run pytest cs336_basics/BPE_Tokenizer/tests
```

the test will train a BPE vocabulary on fixed datasets and serialize the resulting artifacts for inspection.  
You should observe the following folder structure inside `cs336_basics/BPE_Tokenizer/tests/`:

```bash
tests/
├── README.md # This file
├── test_01_train_bpe.py  # The test script of BPE trainer
├── test_01_train_bpe.txt # The test log of BPE trainer
├── test_02_tokenizer.py  # The test script of BPE tokenizer
├── test_02_tokenizer.txt # The test log of BPE tokenizer
├── TinyStoriesV2-GPT4-train-vocab.pkl  # The vocabulary trained on TinyStories 
├── TinyStoriesV2-GPT4-train-merges.pkl # The merges trained on TinyStories
├── owt_train-vocab.pkl  # The vocabulary trained on OpenWebText
└── owt_train-merges.pkl # The merges trained on OpenWebText
```