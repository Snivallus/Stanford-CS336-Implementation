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
import cProfile
import pstats
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
                special_tokens = ["<|endoftext|>"],
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
    def _profile_train_bpe(self, input_path, vocab_size, special_tokens):
        """
        Profile train_bpe on a real (but fixed) dataset to detect
        obvious performance regressions.

        This test:
        - Runs BPE training on TinyStoriesV2-GPT4 or OpenWebText
        - Collects cProfile statistics
        - Asserts basic sanity conditions (non-empty vocab / merges)

        Note:
        This is NOT a strict performance benchmark, but a profiling
        smoke test to surface major bottlenecks during development.
        """
        self.assertTrue(
            os.path.exists(input_path),
            f"Dataset not found at expected path: {input_path}",
        )

        bpe_trainer = BPE_Trainer()

        profiler = cProfile.Profile()
        profiler.enable()

        vocab, merges = bpe_trainer.train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        profiler.disable()

        # --- Basic sanity assertions ---
        self.assertGreater(len(vocab), 256)
        self.assertGreater(len(merges), 0)

        # --- Print profiling summary ---
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.strip_dirs().sort_stats("cumulative").print_stats(25)

        print(
            "\n========== cProfile Results "
            f"({os.path.basename(input_path)}) =========="
        )
        print(s.getvalue())


    def test_02_tinystories_profile_train_bpe(self):
        self._profile_train_bpe(
            input_path = "../datasets/TinyStoriesV2-GPT4-train.txt",
            vocab_size = 256 + 1 + 100, # bytes + <|endoftext|> + small merge budget
            special_tokens=["<|endoftext|>"],
        )


    def test_03_OpenWebText_profile_train_bpe(self):
        self._profile_train_bpe(
            input_path = "../datasets/owt_train.txt",
            vocab_size = 256 + 1 + 100, # bytes + <|endoftext|> + small merge budget
            special_tokens = ["<|endoftext|>"],
        )


if __name__ == "__main__":
    # Redirect unittest output to a log file
    log_path = "cs336_basics/BPE_Tokenizer/tests/train_bpe_test.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    original_stdout = sys.stdout

    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = TeeStdout(original_stdout, f)
        try:
            # Run the unit tests
            unittest.main()
        finally:
            sys.stdout = original_stdout