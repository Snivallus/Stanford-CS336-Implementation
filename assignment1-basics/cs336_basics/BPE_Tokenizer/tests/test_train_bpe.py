"""
Unit and profiling tests for the BPE training implementation.

This file verifies correctness of BPE merges using a canonical
Sennrich et al. (2016) example, and includes lightweight cProfile-based
smoke tests on larger corpora (e.g., TinyStories, OpenWebText) to
surface major performance regressions during development.

All test output is echoed to the console and duplicated to a log file
for offline inspection.
"""

import os
import sys
from io import StringIO
import tempfile
import pickle
import time
import cProfile
import pstats
import psutil
from typing import List
import unittest

from cs336_basics.BPE_Tokenizer.train_bpe import BPE_Trainer # BPE training implementation
from cs336_basics.utils import TeeStdout # Utility to tee stdout to a file

class TestBPETrainingExample(unittest.TestCase):
    """
    Unit test based on the stylized BPE example from
    Sennrich et al. (2016), as described in the assignment.

    This test verifies that the first several merges learned
    by train_bpe exactly match the expected merge sequence.
    """

    def test_01_sennrich_example_merges(self):
        """
        Example corpus:

            low low low low low
            lower lower widest widest widest
            newest newest newest newest newest newest

        Expected first 6 merges:
            ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']
        """

        corpus = (
            "low low low low low\n"
            "lower lower widest widest widest\n"
            "newest newest newest newest newest newest\n"
        )

        expected_merges = [
            (b"s", b"t"),
            (b"e", b"st"),
            (b"o", b"w"),
            (b"l", b"ow"),
            (b"w", b"est"),
            (b"n", b"e"),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(corpus)
            input_path = f.name

        try:
            bpe_trainer = BPE_Trainer()
            vocab, merges = bpe_trainer.train_bpe(
                input_path = input_path,
                vocab_size = 256 + 1 + 6,  # bytes + <|endoftext|> + 6 merges
                special_tokens = ["<|endoftext|>"]
            )

            self.assertGreaterEqual(
                len(merges),
                len(expected_merges),
                "Not enough merges were learned"
            )

            self.assertEqual(
                merges[:6],
                expected_merges,
                "First 6 BPE merges do not match the expected example"
            )

        finally:
            os.remove(input_path)


    # cProfile helper function
    def _profile_train_bpe(
        self, 
        input_path: str, 
        vocab_size: int, 
        special_tokens: List[str], 
        vocab_path: str,
        merges_path: str,
        max_time_in_hour: float = None,
        max_memory_in_GB: float = None,
    ):
        """
        Profile train_bpe on a real dataset to detect obvious performance regressions.

        - Runs BPE training on TinyStoriesV2-GPT4 or OpenWebText
        - Collects cProfile statistics
        - Monitors memory usage and optionally enforces time/memory limits
        """
        self.assertTrue(
            os.path.exists(input_path),
            f"Dataset not found at expected path: {input_path}"
        )

        process = psutil.Process(os.getpid())
        bpe_trainer = BPE_Trainer()

        # --- profile train_bpe ---
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.perf_counter()

        vocab, merges = bpe_trainer.train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )

        end_time = time.perf_counter()
        profiler.disable()

        # --- Serialize results ---
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

        with open(merges_path, "wb") as f:
            pickle.dump(merges, f)

        # --- Longest token diagnostic ---
        longest_token_id, longest_token_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
        try:
            longest_token_str = longest_token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            longest_token_str = "<non-decodable>"

        print(
            "Longest token:\n"
            f"\t-id       = {longest_token_id}\n"
            f"\t-bytes    = {longest_token_bytes!r}\n"
            f"\t-length   = {len(longest_token_bytes)}\n"
            f"\t-utf8_str = {longest_token_str}"
        )

        # --- Memory usage ---
        mem_info = process.memory_info()
        rss_GB = mem_info.rss / 1024**3 # the portion of the process’s memory that is actually loaded in physical RAM
        vms_GB = mem_info.vms / 1024**3 # the total amount of virtual address space the process has access to
        print(f"RSS memory (resident): {rss_GB:.2f} GB")
        print(f"VMS memory (virtual):  {vms_GB:.2f} GB")

        # --- Time and memory checks ---
        elapsed_hours = (end_time - start_time) / 3600
        if max_time_in_hour is not None:
            self.assertLessEqual(
                elapsed_hours, max_time_in_hour,
                f"Training exceeded max_time_in_hour = {max_time_in_hour} h, took {elapsed_hours:.2f} h"
            )

        if max_memory_in_GB is not None:
            self.assertLessEqual(
                rss_GB, max_memory_in_GB,
                f"Training exceeded max_memory_in_GB = {max_memory_in_GB} GB, used {rss_GB:.2f} GB"
            )

        # --- Basic sanity assertions ---
        self.assertGreater(len(vocab), 256 + len(special_tokens))
        self.assertLessEqual(len(vocab), vocab_size)
        self.assertGreater(len(merges), 0)
        self.assertLessEqual(len(merges), vocab_size - 256 - len(special_tokens))

        # --- Print profiling summary ---
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.strip_dirs().sort_stats("cumulative").print_stats(25)

        print(
            f"\n========== cProfile Results ({os.path.basename(input_path)}) =========="
        )
        print(s.getvalue())


    def test_02_tinystories_profile_train_bpe(self):
        self._profile_train_bpe(
            input_path = "../datasets/TinyStoriesV2-GPT4-train.txt",
            vocab_size = 256 + 1 + 9743, # bytes + <|endoftext|> + merge budget
            special_tokens=["<|endoftext|>"],
            vocab_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-vocab.pkl",
            merges_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-merges.pkl", 
            max_time_in_hour = 0.5,
            max_memory_in_GB = 30 
        )


    def test_03_OpenWebText_profile_train_bpe(self):
        self._profile_train_bpe(
            input_path = "../datasets/owt_train.txt",
            vocab_size = 256 + 1 + 31743, # bytes + <|endoftext|> + merge budget
            special_tokens = ["<|endoftext|>"],
            vocab_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-vocab.pkl",
            merges_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-merges.pkl",
            max_time_in_hour = 12,
            max_memory_in_GB = 100 
        )


if __name__ == "__main__":
    # Redirect unittest output to a log file
    log_path = "cs336_basics/BPE_Tokenizer/tests/test_train_bpe.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    original_stdout = sys.stdout

    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = TeeStdout(original_stdout, f)
        try:
            # Run the unit tests
            unittest.main()
        finally:
            sys.stdout = original_stdout