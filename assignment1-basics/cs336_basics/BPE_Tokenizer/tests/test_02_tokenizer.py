import os
import sys
import io
import time
import cProfile
import pstats
import tempfile
from typing import List
import numpy as np
import unittest

from cs336_basics.BPE_Tokenizer import BPE_Tokenizer # implementation of BPE Tokenizer
from cs336_basics.BPE_Tokenizer import (
    TeeStdout, # Utility to tee stdout to a file
    sample_documents # Utility to sample documents from a directory
)


LOG_PATH = "cs336_basics/BPE_Tokenizer/tests/test_02_tokenizer.txt"
SEED = 51
NUM_SAMPLED_DOCS = 10
MIN_DOC_SIZE = 1 * 1024 * 1024 # 1 MB


class TestBPETokenizer(unittest.TestCase):
    """
    Unit tests for BPE Tokenizer
    """
    @classmethod
    def setUpClass(cls):
        """
        class setup: tee stdout to a log file
        """
        cls.log_path = LOG_PATH
        os.makedirs(os.path.dirname(cls.log_path), exist_ok=True)
        cls._orig_stdout = sys.stdout
        cls._log_file = open(cls.log_path, "w", encoding="utf-8")
        sys.stdout = TeeStdout(sys.stdout, cls._log_file)

    @classmethod
    def tearDownClass(cls):
        """
        class teardown: restore stdout and close log file
        """
        sys.stdout = cls._orig_stdout
        cls._log_file.close()


    def test_01_bpe_encoding_example(self):
        """
        Input: "the cat ate"

        Expected encoding:
        [9, 7, 1, 5, 10, 3]
        """
        print(f"\n[{self._testMethodName}]")
        vocab = {
            0: b" ",
            1: b"a",
            2: b"c",
            3: b"e",
            4: b"h",
            5: b"t",
            6: b"th",
            7: b" c",
            8: b" a",
            9: b"the",
            10: b" at",
        }

        merges = [
            (b"t", b"h"),
            (b" ", b"c"),
            (b" ", b"a"),
            (b"th", b"e"),
            (b" a", b"t"),
        ]

        special_tokens = []

        tokenizer = BPE_Tokenizer(
            vocab = vocab,
            merges = merges,
            special_tokens = special_tokens,
        )

        text = b"the cat ate"
        encoded = tokenizer.encode(text)

        expected_ids = [9, 7, 1, 5, 10, 3]

        print(f"Encoded IDs: {encoded}")
        self.assertEqual(
            encoded,
            expected_ids,
            msg = "BPE encoding does not match expected output",
        )

        decoded = tokenizer.decode(encoded)

        print(f"Decoded text: {decoded}")
        self.assertEqual(
            decoded,
            "the cat ate",
            msg = "Decoded text does not match original input",
        )


    def _profile_tokenizer(
        self, 
        vocab_path: str,
        merges_path: str,
        special_tokens: List[str],
        dataset_path: str,
        token_ids_path: str,
        max_num_workers: int,
        chunk_per_worker: int,
        min_chunk_size: int,
        profile_decode: bool = True,
    ):
        """
        End-to-end validation + profiling of a trained BPE tokenizer.

        Steps:
        1. Encode/decode first 100 characters and compare strings
        2. Sample 10 documents and measure average compression ratio (bytes/token)
        3. Measure throughput on the full dataset
        4. Profile encode(path) on the full dataset
        5. Profile decode(token_ids) on the full dataset
        """

        # --- Load tokenizer ---
        tokenizer = BPE_Tokenizer.from_files(
            vocab_path = vocab_path,
            merges_path = merges_path,
            special_tokens = special_tokens,
        )

        self.assertTrue(
            os.path.exists(dataset_path),
            f"Validation dataset not found: {dataset_path}"
        )

        # ============================================================
        # 1. Local correctness check (first 100 characters)
        # ============================================================
        with open(dataset_path, "r", encoding="utf-8") as f:
            first_100_chars = f.read(100)

        self.assertGreater(
            len(first_100_chars), 0,
            "Validation file is empty"
        )

        encoded_100 = tokenizer.encode(
            input = first_100_chars,
            max_num_workers = max_num_workers,
            chunk_per_worker = chunk_per_worker,
            min_chunk_size = min_chunk_size
        )
        decoded_100 = tokenizer.decode(encoded_100)

        self.assertEqual(
            decoded_100,
            first_100_chars,
            "Encode → decode mismatch on first 100 characters"
        )

        print("[OK] First-100-character encode/decode check passed")

        # ============================================================
        # 2. Sample 10 documents and compute compression ratio
        # ============================================================
        sampled_docs = sample_documents(
            file_path = dataset_path, 
            split_special_token = b"<|endoftext|>",
            num_samples = NUM_SAMPLED_DOCS,
            desired_num_chunks = 10 * NUM_SAMPLED_DOCS,
            min_doc_size = MIN_DOC_SIZE,
            seed = SEED
        )

        total_bytes = 0
        total_tokens = 0

        for doc in sampled_docs:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
                    tmp.write(doc)
                    tmp.flush()
                    tmp_path = tmp.name

                ids = tokenizer.encode(
                    input = tmp_path,
                    max_num_workers = max_num_workers,
                    chunk_per_worker = chunk_per_worker,
                    min_chunk_size = min_chunk_size
                )

                total_bytes += len(doc.encode("utf-8"))
                total_tokens += len(ids)

            finally:
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        compression_ratio = total_bytes / total_tokens

        # ============================================================
        # 3. Profile encode(path)
        # ============================================================
        encode_profiler = cProfile.Profile()
        encode_profiler.enable()
        start_time = time.perf_counter()
        token_ids = tokenizer.encode(
            input = dataset_path,
            max_num_workers = max_num_workers,
            chunk_per_worker = chunk_per_worker,
            min_chunk_size = min_chunk_size
        )
        end_time = time.perf_counter()
        encode_profiler.disable()

        encode_stats_stream = io.StringIO()
        encode_stats = pstats.Stats(encode_profiler, stream=encode_stats_stream)
        encode_stats.strip_dirs().sort_stats("cumtime").print_stats(25)

        elapsed = end_time - start_time
        file_bytes = os.path.getsize(dataset_path)
        throughput = (file_bytes / elapsed) / (1024 * 1024) # MB/sec

        print(
            "[OK] Calculating statistics of tokenizer:\n"
            f"  - vocab size : {len(tokenizer.vocab)}\n"
            f"  - merges     : {len(tokenizer.merges)}\n"
            f"  - bytes in file: {file_bytes}\n"
            f"  - token count: {len(token_ids)}\n"
            f"  - compression ratio (avg over {NUM_SAMPLED_DOCS} docs, seed = {SEED}): {compression_ratio:.4f} bytes/token\n"
            f"  - elapsed time: {elapsed:.2f} sec\n"
            f"  - throughput: {throughput:.2f} MB/sec"
        )

        # save token_ids as a Numpy array of datatype `uint16`
        token_ids_np = np.array(token_ids, dtype=np.uint16)
        np.save(token_ids_path, token_ids_np, allow_pickle=False)

        print("\n========== ENCODE PROFILING (top 25 by cumulative time) ==========")
        print(encode_stats_stream.getvalue())

        # ============================================================
        # 4. Profile decode(token_ids)
        # ============================================================
        if profile_decode:
            decode_profiler = cProfile.Profile()
            decode_profiler.enable()
            tokenizer.decode(token_ids)
            decode_profiler.disable()

            decode_stats_stream = io.StringIO()
            decode_stats = pstats.Stats(decode_profiler, stream=decode_stats_stream)
            decode_stats.strip_dirs().sort_stats("cumtime").print_stats(25)

            print("\n========== DECODE PROFILING (top 25 by cumulative time) ==========")
            print(decode_stats_stream.getvalue())


    def test_02_tinystories_valid_profile_tokenizer(self):
        print(f"\n[{self._testMethodName}]")
        self._profile_tokenizer(
            vocab_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-vocab.pkl",
            merges_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-merges.pkl",
            special_tokens = ["<|endoftext|>"], 
            dataset_path = "../datasets/TinyStoriesV2-GPT4-valid.txt",
            token_ids_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-valid.npy",
            max_num_workers = 64,
            chunk_per_worker = 1,
            min_chunk_size = 256 * 1024, # 256 KB
            profile_decode = True
        )

    
    def test_03_tinystories_train_profile_tokenizer(self):
        print(f"\n[{self._testMethodName}]")
        self._profile_tokenizer(
            vocab_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-vocab.pkl",
            merges_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-merges.pkl",
            special_tokens = ["<|endoftext|>"], 
            dataset_path = "../datasets/TinyStoriesV2-GPT4-train.txt",
            token_ids_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train.npy", 
            max_num_workers = 64,
            chunk_per_worker = 4,
            min_chunk_size = 1024 * 1024, # 1 MB
            profile_decode = False
        )


    def test_04_OpenWebText_with_TinyStories_tokenizer(self):
        """
        Experiment: Tokenize OpenWebText sample using TinyStories tokenizer
        Compare compression ratio with TinyStories documents and provide qualitative description
        """
        print(f"\n[{self._testMethodName}]")

        tiny_vocab_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-vocab.pkl"
        tiny_merges_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-merges.pkl"
        special_tokens = ["<|endoftext|>"]
        owt_valid_path = "../datasets/owt_valid.txt"
        TinyStories_valid_path = "../datasets/TinyStoriesV2-GPT4-valid.txt"

        # Load TinyStories tokenizer
        tokenizer = BPE_Tokenizer.from_files(
            vocab_path = tiny_vocab_path,
            merges_path = tiny_merges_path,
            special_tokens = special_tokens,
        )

        def compute_avg_compression_ratio(docs):
            total_bytes = 0
            total_tokens = 0
            for doc in docs:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
                        tmp.write(doc)
                        tmp.flush()
                        tmp_path = tmp.name

                    ids = tokenizer.encode(tmp_path)
                    total_bytes += len(doc.encode("utf-8"))
                    total_tokens += len(ids)
                finally:
                    if tmp_path is not None and os.path.exists(tmp_path):
                        os.remove(tmp_path)
            return total_bytes, total_tokens, total_bytes / total_tokens

        # Sample 10 documents each
        owt_sampled_docs = sample_documents(
            file_path = owt_valid_path, 
            split_special_token = b"<|endoftext|>",
            num_samples = NUM_SAMPLED_DOCS,
            desired_num_chunks = 10 * NUM_SAMPLED_DOCS,
            min_doc_size = MIN_DOC_SIZE,
            seed = SEED
        )

        tiny_sampled_docs = sample_documents(
            file_path = TinyStories_valid_path, 
            split_special_token = b"<|endoftext|>",
            num_samples = NUM_SAMPLED_DOCS,
            desired_num_chunks = 10 * NUM_SAMPLED_DOCS,
            min_doc_size = MIN_DOC_SIZE,
            seed = SEED
        )

        # Compute compression ratios
        owt_total_bytes, owt_total_tokens, owt_compression = compute_avg_compression_ratio(owt_sampled_docs)
        tiny_total_bytes, tiny_total_tokens, tiny_compression = compute_avg_compression_ratio(tiny_sampled_docs)

        print(
            "[Experiment] OpenWebText encoded with TinyStories tokenizer\n"
            f"  - total bytes : {owt_total_bytes}\n"
            f"  - total tokens : {owt_total_tokens}\n"
            f"  - avg bytes/token : {owt_compression:.4f}"
        )

        print(
            "\n[Experiment] TinyStories encoded with TinyStories tokenizer\n"
            f"  - total bytes : {tiny_total_bytes}\n"
            f"  - total tokens : {tiny_total_tokens}\n"
            f"  - avg bytes/token : {tiny_compression:.4f}"
        )


    def test_05_OpenWebText_valid_profile_tokenizer(self):
        print(f"\n[{self._testMethodName}]")
        self._profile_tokenizer(
            vocab_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-vocab.pkl",
            merges_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-merges.pkl",
            special_tokens = ["<|endoftext|>"], 
            dataset_path = "../datasets/owt_valid.txt",
            token_ids_path = "cs336_basics/BPE_Tokenizer/tests/owt_valid.npy",
            max_num_workers = 64,
            chunk_per_worker = 2,
            min_chunk_size = 1024 * 1024, # 1 MB
            profile_decode = True
        )


    def test_06_OpenWebText_train_profile_tokenizer(self):
        print(f"\n[{self._testMethodName}]")
        self._profile_tokenizer(
            vocab_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-vocab.pkl",
            merges_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-merges.pkl",
            special_tokens = ["<|endoftext|>"], 
            dataset_path = "../datasets/owt_train.txt",
            token_ids_path = "cs336_basics/BPE_Tokenizer/tests/owt_train.npy",
            max_num_workers = 64,
            chunk_per_worker = 8,
            min_chunk_size = 1024 * 1024, # 1 MB
            profile_decode = False
        )


if __name__ == "__main__":
    # Redirect unittest output to a log file
    log_path = LOG_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    original_stdout = sys.stdout

    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = TeeStdout(original_stdout, f)
        try:
            # Run the unit tests
            unittest.main()
        finally:
            sys.stdout = original_stdout