import os
import sys
import io
import cProfile
import pstats
from typing import List
import unittest

from cs336_basics.BPE_Tokenizer import BPE_Tokenizer # implementation of BPE Tokenizer
from cs336_basics.utils import TeeStdout # Utility to tee stdout to a file


LOG_PATH = "cs336_basics/BPE_Tokenizer/tests/test_02_tokenizer.txt"


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

        print(f"\nEncoded IDs: {encoded}")
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
        validation_path: str,
    ):
        """
        End-to-end validation + profiling of a trained BPE tokenizer.

        Steps:
        1. Encode/decode first 100 characters and compare strings
        2. Profile encode(path) on the full dataset
        3. Profile decode(token_ids) on the full dataset
        """

        # --- Load tokenizer ---
        tokenizer = BPE_Tokenizer.from_files(
            vocab_filepath = vocab_path,
            merges_filepath = merges_path,
            special_tokens = special_tokens,
        )

        self.assertTrue(
            os.path.exists(validation_path),
            f"Validation dataset not found: {validation_path}"
        )

        # ============================================================
        # 1. Local correctness check (first 100 characters)
        # ============================================================
        with open(validation_path, "r", encoding="utf-8") as f:
            first_100_chars = f.read(100)

        self.assertGreater(
            len(first_100_chars), 0,
            "Validation file is empty"
        )

        encoded_100 = tokenizer.encode(first_100_chars)
        decoded_100 = tokenizer.decode(encoded_100)

        self.assertEqual(
            decoded_100,
            first_100_chars,
            "Encode → decode mismatch on first 100 characters"
        )

        print("\n[OK] First-100-character encode/decode check passed")

        # ============================================================
        # 2. Encode full dataset (needed for profiling decode)
        # ============================================================
        token_ids = tokenizer.encode(validation_path)

        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 0)

        # ============================================================
        # 3. Profile encode(path)
        # ============================================================
        encode_profiler = cProfile.Profile()
        encode_profiler.enable()
        tokenizer.encode(validation_path)
        encode_profiler.disable()

        encode_stats_stream = io.StringIO()
        encode_stats = pstats.Stats(encode_profiler, stream=encode_stats_stream)
        encode_stats.strip_dirs().sort_stats("cumtime").print_stats(25)

        print("\n========== ENCODE PROFILING (top 25 by cumulative time) ==========")
        print(encode_stats_stream.getvalue())

        # ============================================================
        # 4. Profile decode(token_ids)
        # ============================================================
        decode_profiler = cProfile.Profile()
        decode_profiler.enable()
        tokenizer.decode(token_ids)
        decode_profiler.disable()

        decode_stats_stream = io.StringIO()
        decode_stats = pstats.Stats(decode_profiler, stream=decode_stats_stream)
        decode_stats.strip_dirs().sort_stats("cumtime").print_stats(25)

        print("\n========== DECODE PROFILING (top 25 by cumulative time) ==========")
        print(decode_stats_stream.getvalue())

        print(
            "[OK] Tokenizer profiling complete\n"
            f"  - vocab size : {len(tokenizer.vocab)}\n"
            f"  - merges     : {len(tokenizer.merges)}\n"
            f"  - token count: {len(token_ids)}\n"
        )


    def test_02_tinystories_profile_tokenizer(self):
        self._profile_tokenizer(
            vocab_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-vocab.pkl",
            merges_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-merges.pkl",
            special_tokens = ["<|endoftext|>"], 
            validation_path = "../datasets/TinyStoriesV2-GPT4-valid.txt"
        )


    def test_03_OpenWebText_profile_tokenizer(self):
        self._profile_tokenizer(
            vocab_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-vocab.pkl",
            merges_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-merges.pkl",
            special_tokens = ["<|endoftext|>"], 
            validation_path = "../datasets/owt_valid.txt"
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