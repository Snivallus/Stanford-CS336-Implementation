import os
from io import StringIO
import tempfile
import cProfile
import pstats
import unittest

from cs336_basics.BPE_Tokenizer.train_bpe import BPE_Trainer


class TestBPETrainingExample(unittest.TestCase):
    """
    Unit test based on the stylized BPE example from
    Sennrich et al. (2016), as described in the assignment.

    This test verifies that the first several merges learned
    by train_bpe exactly match the expected merge sequence.
    """

    def test_sennrich_example_merges(self):
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
                input_path=input_path,
                vocab_size=256 + 1 + 6,  # bytes + <|endoftext|> + 6 merges
                special_tokens=["<|endoftext|>"],
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


    def test_tinystories_valid_profile_train_bpe(self):
        """
        Profile train_bpe on a real (but fixed) dataset to detect
        obvious performance regressions.

        This test:
        - Runs BPE training on TinyStoriesV2-GPT4-valid.txt
        - Collects cProfile statistics
        - Asserts basic sanity conditions (non-empty vocab / merges)

        Note:
        This is NOT a strict performance benchmark, but a profiling
        smoke test to surface major bottlenecks during development.
        """

        input_path = "../datasets/TinyStoriesV2-GPT4-train.txt"

        self.assertTrue(
            os.path.exists(input_path),
            f"Dataset not found at expected path: {input_path}",
        )

        bpe_trainer = BPE_Trainer()

        profiler = cProfile.Profile()
        profiler.enable()

        vocab, merges = bpe_trainer.train_bpe(
            input_path=input_path,
            vocab_size=256 + 1 + 100,  # bytes + <|endoftext|> + small merge budget
            special_tokens=["<|endoftext|>"],
        )

        profiler.disable()

        # --- Basic sanity assertions ---
        self.assertGreater(
            len(vocab),
            256,
            "Vocabulary should grow beyond raw bytes",
        )

        self.assertGreater(
            len(merges),
            0,
            "Expected at least one BPE merge to be learned",
        )

        # --- Print profiling summary (top cumulative time) ---
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.strip_dirs().sort_stats("cumulative").print_stats(25)

        print("\n========== cProfile Results (Top 25 by cumulative time) ==========")
        print(s.getvalue())


if __name__ == "__main__":
    unittest.main()