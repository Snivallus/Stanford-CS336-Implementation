# BPE Tokenizer — Test Artifacts

After running the BPE training test from the **project root**:

```bash
python cs336_basics/BPE_Tokenizer/tests/train_bpe_test.py
```

the test will train a BPE vocabulary on fixed datasets and serialize the resulting artifacts for inspection.  
You should observe the following folder structure inside `cs336_basics/BPE_Tokenizer/tests/`:

```bash
tests/
├── README.md # This file
├── train_bpe_test.py  # The test script
├── train_bpe_test.txt # The test log
├── TinyStoriesV2-GPT4-train-vocab.pkl  # The vocabulary trained on TinyStories 
├── TinyStoriesV2-GPT4-train-merges.pkl # The merges trained on TinyStories
├── owt_train-vocab.pkl  # The vocabulary trained on OpenWebText
└── owt_train-merges.pkl # The merges trained on OpenWebText
```