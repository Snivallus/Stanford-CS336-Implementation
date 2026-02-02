"""
Byte Pair Encoding (BPE) trainer.

Features:
- GPT-2 style regex pretokenization
- Parallel word counting with memory-mapped I/O
- Iterative merge of most frequent byte/token pairs using a lazy max-heap
- Optional C/Cython acceleration for pair merging hot loops
"""

import os
import mmap
import gc
import regex as re
from collections import defaultdict, Counter
import multiprocessing as mp
import time
import argparse
from typing import (
    BinaryIO,
    List,
    Dict,
    Tuple,
    DefaultDict  
)

Word = bytes
TokenId = int
Pair = Tuple[TokenId, TokenId]

WordEncoding = List[TokenId] # list of token IDs representing a word
Vocab = Dict[TokenId, bytes]
WordCounter = Counter[Word]
PairCounter = DefaultDict[Pair, int]
PairStrings = Dict[Pair, Tuple[bytes, bytes]]  # map from token ID pair to byte string representation
PairToWords = DefaultDict[Pair, Counter[Word]] # map from pair to affected words with occurrence counts

import cs336_basics.BPE_Tokenizer.max_heapq as maxheap
# try to import CPython native implementation
try:
    from cs336_basics.BPE_Tokenizer.bpe_cpython import _merge_pair_and_count_pair_difference
    _HAS_NATIVE = True
except ImportError:
    print("""
            Failed to import CPython native implementation of _merge_pair_and_count_pair_difference, 
            Fallback to its Python implementation.
    """)
    _HAS_NATIVE = False


MIN_CHUNK_SIZE = 4 * 1024 * 1024 # 4 MB
N_BYTES = 256
MAX_NUM_COUNTERS = 64

class BPE_Trainer():
    def train_bpe(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
        *args: str,
    ) -> Tuple[Vocab, List[Tuple[bytes, bytes]]]:
        """
        Train a Byte-Pair Encoding (BPE) tokenizer on a given dataset.

        This method performs the full BPE training pipeline:
        1. Pre-tokenizes the input text into words using GPT-2 style regex
           and counts their frequencies in parallel using multiple processes.
        2. Initializes the vocabulary with single-byte tokens and any 
           user-specified special tokens.
        3. Encodes each unique word as a sequence of token IDs (byte-level).
        4. Counts all adjacent token pairs in the corpus and stores both
           the pair frequencies and their mapping to affected words.
        5. Builds a max-heap of token pairs by frequency to efficiently 
           select the most frequent pair for merging.
        6. Iteratively merges the most frequent pair, updating word encodings,
           pair counts, and the vocabulary, until the target vocab size is reached.

        Parameters:
        - input_path (str): Path to the training dataset (raw text or binary).
        - vocab_size (int): Desired final vocabulary size after BPE merges.
        - special_tokens (List[str]): List of special tokens to include in the vocabulary.
        - *args (str): Optional command-line arguments, currently supports:
            - "--max_num_counters" / "-c": maximum number of parallel counting processes.

        Returns:
        - vocab (Vocab): Dictionary mapping token IDs to their byte string representation.
        - merges (List[Tuple[bytes, bytes]]): List of byte pair merges applied during training.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--max_num_counters", 
            "-c",                
            type = int, 
            default = MAX_NUM_COUNTERS, 
            help = "maximum number of parallel counting processes"
        )

        args = parser.parse_args(args)
        print(f"train_bpe: {args = }")
        max_num_counters = args.max_num_counters

        # pretokenize and count words
        start_time = time.perf_counter()
        word_counts = self._pretokenize_and_count_mp(
            input_path, 
            special_tokens, 
            max_num_counters
        )
        end_time = time.perf_counter()
        print(f"_pretokenize_and_count_words: {end_time - start_time} sec for {len(word_counts)} unique words")
        
        # initialize vocabulary
        vocab = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            vocab[N_BYTES + i] = token.encode('utf-8')
        size = N_BYTES + len(special_tokens)
        merges = []

        # encode words to byte ids
        word_encodings = {} # hash map from word to list of byte ids
        for word in word_counts:
            word_encodings[word] = list(word)

        # count initial pairs
        pair_strings = {} # hash map from pair to its string representation (tuple of strings)
        pair_to_words = defaultdict(Counter) # hash map from pair to words containing the pair (with occurrence count of the pair in each word)
        start_time = time.perf_counter()
        pair_counts = self._count_pairs(
            word_counts, 
            word_encodings, 
            pair_strings, 
            vocab, 
            pair_to_words
        )
        end_time = time.perf_counter()
        print(f"_count_pairs: {end_time - start_time:.2f} sec for {len(pair_counts)} unique pairs")

        # build maxheap
        start_time = time.perf_counter()
        pair_heap = []
        for pair, count in pair_counts.items():
            maxheap.heappush_max(
                pair_heap, 
                (count, pair_strings[pair], pair)
            )
        end_time = time.perf_counter()
        print(f"_build_heap: {end_time - start_time:.2f} sec for {len(pair_heap)} heap size")

        # perform merges
        start_time = time.perf_counter()
        while size < vocab_size:
            self._merge_a_pair(
                pair_counts, 
                pair_strings, 
                vocab,
                pair_to_words, 
                word_counts, 
                word_encodings,
                merges, 
                size, 
                pair_heap
            )
            size += 1
        end_time = time.perf_counter()
        print(f"_merge: {end_time - start_time} sec for {vocab_size - (N_BYTES + len(special_tokens))} merges\n")               
        
        return vocab, merges


    def _pretokenize_and_count_mp(
        self,
        input_path: str,
        special_tokens: List[str],
        max_num_counters: int,
    ) -> WordCounter:
        """
        Pre-tokenize the input text and count word frequencies in parallel.

        This method performs word-level tokenization using GPT-2 style regex and
        counts occurrences across the corpus using multiple processes. It supports
        memory-efficient access via memory-mapped files and optional handling of 
        special tokens.

        Steps:
        1. Compile GPT-2 regex for standard word tokenization.
        2. Compile regex for user-specified special tokens.
        3. Memory-map the input file for efficient concurrent access.
        4. Split the file into chunks aligned with a special token boundary.
        5. Use multiprocessing to count words in each chunk independently.
        6. Aggregate the results into a single WordCounter.

        Parameters:
        - input_path (str): Path to the input text or binary file.
        - special_tokens (List[str]): List of special tokens to recognize as separate words.
        - max_num_counters (int): Maximum number of parallel counting processes.

        Returns:
        - WordCounter: Counter object mapping each word (bytes) to its frequency.
        """

        # GPT-2 regex
        gpt2_pattern = re.compile(
            rb"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"""
        )

        # build split pattern
        special_token_pattern = b"|".join(
            re.escape(token.encode("utf-8")) for token in special_tokens
        )
        special_token_pattern = re.compile(special_token_pattern)

        # memory-map the file for shared access
        with open(input_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # find chunk boundaries
            boundaries = self.find_chunk_boundaries(
                file = f,
                desired_num_chunks = max_num_counters,
                split_special_token = b"<|endoftext|>",
            )

            # create chunk ranges
            ranges = [
                (input_path, boundaries[i], boundaries[i+1], gpt2_pattern, special_token_pattern)
                for i in range(len(boundaries) - 1)
            ]

            # count in parallel and aggregate results
            word_counts = Counter()

            # leave half CPUs free for OS and I/O
            num_procs = min(
                max(1, os.cpu_count() // 2),
                len(ranges),
            )

            with mp.Pool(processes=num_procs) as pool:
                # use imap_unordered to consume results as workers finish
                for c in pool.imap_unordered(
                    self._count_chunk_process,
                    ranges,
                    chunksize=1,
                ):
                    # aggregate partial counters
                    word_counts.update(c)

            mm.close() # close memory-mapped file
            gc.collect() # force garbage collection to free memory

        return word_counts


    @staticmethod
    def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> List[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = MIN_CHUNK_SIZE  # Read ahead by MIN_CHUNK_SIZE bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


    @staticmethod
    def _count_chunk_process(
        args: Tuple[str, int, int, re.Pattern[bytes], re.Pattern[bytes]],
    ) -> Counter[bytes]:
        """
        Count word occurrences in a chunk of the dataset.

        This function is intended to be run in parallel by multiple processes
        to compute word frequencies over independent portions of the corpus.

        Steps:
        1. Memory-map the input file to efficiently access the specified chunk.
        2. Slice the chunk from the start to end byte offsets.
        3. Split the chunk into blocks using special tokens as separators.
        4. Apply GPT-2 style regex to each block to extract words/tokens.
        5. Count occurrences of each token in the chunk using a Counter.

        Parameters:
        - args: Tuple containing:
            - input_path (str): Path to the dataset file.
            - start (int): Start byte index of the chunk.
            - end (int): End byte index of the chunk.
            - gpt2_pattern (re.Pattern[bytes]): Regex for GPT-2 style tokenization.
            - special_token_pattern (re.Pattern[bytes]): Regex for special tokens.

        Returns:
        - Counter[bytes]: Frequency count of all words/tokens in the chunk.
        """

        # unpack args
        input_path, start, end, gpt2_pattern, special_token_pattern = args

        # split data by special tokens and count in each block
        counter = Counter()
        with open(input_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            data = mm[start:end]

            blocks = special_token_pattern.split(data)
            for block in blocks:
                for m in gpt2_pattern.finditer(block):
                    counter[m.group(0)] += 1

            mm.close()

        return counter


    @staticmethod
    def _count_pairs(
        word_counts: WordCounter,
        word_encodings: Dict[Word, WordEncoding],
        pair_strings: PairStrings,
        vocab: Vocab,
        pair_to_words: PairToWords,
    ) -> PairCounter:
        """
        Count all adjacent token pairs in the corpus and track their occurrences.

        This function computes:
        1. The total frequency of each token pair across all words (corpus-level).
        2. The occurrence of each pair within individual words (word-level structure).
        3. Stores the byte-string representation of each pair for later use in merges.

        Parameters:
        - word_counts: Counter of each unique word (bytes) and its frequency.
        - word_encodings: Mapping of each word to its current sequence of token IDs.
        - pair_strings: Mapping from each token pair to its byte representation.
        - vocab: Mapping of token IDs to their byte values.
        - pair_to_words: Nested dict mapping each pair to words containing
        it and the occurrence count within that word.

        Returns:
        - PairCounter: Mapping from token pairs (tuple of token IDs) to their total
        frequency in the corpus.
        """

        # Initialize a counter for the frequency of each pair in the entire corpus
        pair_counts = defaultdict(int) 
        
        # count pairs in all words
        for word, count in word_counts.items():
            # Retrieve the current token ID encoding for the word
            encoding = word_encodings[word]
            
            # Iterate over all adjacent token pairs in the word
            for i in range(0, len(encoding) - 1):
                pair = encoding[i], encoding[i+1]
                
                # Increment the global pair count by the frequency of the word
                pair_counts[pair] += count

                # Increment the word-level count for this pair
                pair_to_words[pair][word] += 1

                # Store the byte representation of the pair if not already recorded
                if pair not in pair_strings:
                    pair_strings[pair] = (vocab[pair[0]], vocab[pair[1]])
        
        return pair_counts


    def _merge_a_pair(
        self,
        pair_counts: PairCounter,
        pair_strings: PairStrings,
        vocab: Vocab,
        pair_to_words: PairToWords,
        word_counts: WordCounter,
        word_encodings: Dict[Word, WordEncoding],
        merges: List[Tuple[bytes, bytes]],
        size: int,
        pair_heap: List[Tuple[int, Tuple[bytes, bytes], Pair]],
    ) -> None:
        """
        Merge the most frequent token pair in the vocabulary and update all relevant structures.

        This function performs one iteration of the BPE merge process:
        1. Selects the most frequent pair from the max-heap.
        2. Validates that the pair count is current; if outdated, reinsert with updated count.
        3. Merges the pair into a new token in the vocabulary.
        4. Updates all affected word encodings, pair counts, and pair-to-word mappings.
        5. Records the merge in the merges list for later use.

        Parameters:
        - pair_counts: Corpus-level frequency of each token pair.
        - pair_strings: Byte string representation of each token pair.
        - vocab: Mapping of token IDs to byte strings.
        - pair_to_words: Mapping of each pair to words containing it with occurrence counts.
        - word_counts: Frequency of each word in the corpus.
        - word_encodings: Current token ID sequence for each word.
        - merges: List of all merges performed so far.
        - size: Current vocabulary size; used as the new token ID for the merged pair.
        - pair_heap: Max-heap of pairs sorted by frequency.

        Notes:
        - Uses a lazy max-heap to efficiently select the most frequent pair.
        - If the heap contains outdated counts, the function reinserts updated counts instead of rebuilding.
        - Affected word encodings are updated in-place, and all related pair counts and mappings are incrementally adjusted.
        """

        # Select the most frequent pair from the max-heap
        while pair_heap:
            count, string_priority, pair_to_be_merged = maxheap.heappop_max(pair_heap)

            # Verify that the pair still exists in pair_counts
            if pair_to_be_merged in pair_counts:
                recorded_count = pair_counts[pair_to_be_merged]

                # Check if the count matches the heap (lazy heap validation)
                if recorded_count == count:
                    # Found valid pair to merge
                    break
                else:
                    # Outdated count, push back updated count into heap
                    maxheap.heappush_max(
                        pair_heap,
                        (recorded_count, string_priority, pair_to_be_merged)
                    )
        else: # no break => merge_pair not found (should not happen)
            Warning("no valid pairs found!")
            return False

        # Merge the selected pair into a new token
        merged_bytes = vocab[pair_to_be_merged[0]] + vocab[pair_to_be_merged[1]]
        vocab[size] = merged_bytes  # Add new token to vocabulary
        new_id = size  # Assign new token ID

        # Get all words affected by this pair
        affected_words = pair_to_words[pair_to_be_merged].keys()
        
        # Update encodings, pair counts, and pair-to-word mappings for affected words
        self._update_pair_count_of_affected_words(
            pair_to_be_merged, 
            affected_words, 
            word_encodings,
            word_counts, 
            pair_counts,
            pair_to_words, 
            new_id, 
            pair_strings, 
            vocab, 
            pair_heap
        )

        # Record the merge for later reconstruction
        merges.append((vocab[pair_to_be_merged[0]], vocab[pair_to_be_merged[1]]))


    def _update_pair_count_of_affected_words(
        self,
        pair_to_be_merged: Pair,
        affected_words: Counter[Word],
        word_encodings: Dict[Word, WordEncoding],
        word_counts: WordCounter,
        pair_counts: PairCounter,
        pair_to_words: PairToWords,
        new_id: TokenId,
        pair_strings: PairStrings,
        vocab: Vocab,
        pair_heap: List[Tuple[int, Tuple[bytes, bytes], Pair]],
    ) -> None:
        # hash map from pair to its cumulative change in the corpus after merging;
        # applied globally after processing all affected words
        pair_count_difference = defaultdict(int)
        
        # break down pair_to_be_merged into its constituent bytes
        bytes_a, bytes_b = pair_to_be_merged
        
        for word in affected_words:
            # Skip invalid words whose pair occurrence count is non-positive.
            # We use lazy deletion here: entries with zero or negative counts
            # are left in pair_to_words and filtered out at access time to avoid
            # expensive dict deletions during merging .
            if pair_to_words[pair_to_be_merged][word] <= 0:
                continue
            
            # get affected tokens and word counts
            old_encoding = word_encodings[word]
            word_freq = word_counts[word]

            # merge pair and get pair count differences
            old_pairs, new_encoding, new_pairs = self._merge_pair_and_count_pair_difference(
                old_encoding, 
                bytes_a,
                bytes_b,
                new_id
            )

            # Remove old pair contributions
            for a, b, k in old_pairs:
                pair = (a, b)
                pair_to_words[pair][word] -= k # lazy deletion (do not eagerly remove zero- or negative-count entries)
                pair_count_difference[pair] -= k * word_freq

            # Update word encoding
            word_encodings[word] = new_encoding

            # Add new pair contributions
            for a, b, k in new_pairs:
                pair = (a, b)
                pair_to_words[pair][word] += k
                pair_count_difference[pair] += k * word_freq

        # Apply global pair count differences
        for pair, diff in pair_count_difference.items():
            if diff == 0:
                continue

            new_count = pair_counts[pair] + diff

            if new_count > 0:
                pair_counts[pair] = new_count

                # record pair string if first seen
                if pair not in pair_strings:
                    pair_strings[pair] = (vocab[pair[0]], vocab[pair[1]])

                # lazy heap update
                maxheap.heappush_max(
                    pair_heap,
                    (new_count, pair_strings[pair], pair),
                )
            else:
                # pair completely removed
                pair_counts.pop(pair, None)
                pair_to_words.pop(pair, None)


    @staticmethod
    def _merge_pair_and_count_pair_difference(
        old_encoding: list[int],
        bytes_a: int,
        bytes_b: int,
        new_id: int,
    ) -> tuple[list[int], dict[tuple[int, int], int]]:
        """
        Merge a specific token pair in a single word encoding and compute the changes in pair counts.

        This function performs the core token-level merge operation for BPE training:
        1. Counts the frequency of all adjacent pairs in the original word (`old_encoding`).
        2. Merges occurrences of the target pair `(bytes_a, bytes_b)` into a new token ID (`new_id`).
        3. Computes the frequency of all adjacent pairs in the updated word (`new_encoding`).
        4. Returns the old pair counts, the updated encoding, and the new pair counts for downstream updates.

        Parameters:
        - old_encoding: The current token ID sequence for a single word.
        - bytes_a: The first token ID in the pair to be merged.
        - bytes_b: The second token ID in the pair to be merged.
        - new_id: The token ID representing the merged pair.

        Returns:
        - old_pairs (list[tuple[int, int, int]]): List of (token_a, token_b, count) for all pairs in the original word.
        - new_encoding (list[int]): Updated token ID sequence after merging the pair.
        - new_pairs (list[tuple[int, int, int]]): List of (token_a, token_b, count) for all pairs in the updated word.

        Notes:
        - If a high-performance CPython implementation is available (_HAS_NATIVE), it is used instead.
        - The function produces a "flat" list of pairs with counts to simplify integration with the main BPE trainer logic.
        """

        # If CPython implementation is available, use it;
        # Otherwise, fallback to Python implementation.
        if _HAS_NATIVE:
            return _merge_pair_and_count_pair_difference(
                old_encoding, 
                bytes_a, 
                bytes_b, 
                new_id
            )

        # count old pairs in this word
        old_pair_counter = Counter()
        for i in range(len(old_encoding) - 1):
            old_pair_counter[(old_encoding[i], old_encoding[i+1])] += 1

        # merge pair_to_be_merged
        new_encoding = []
        i = 0
        while i < len(old_encoding):
            if (
                i < len(old_encoding) - 1
                and old_encoding[i] == bytes_a
                and old_encoding[i+1] == bytes_b
            ):
                # merge
                new_encoding.append(new_id)
                i += 2
            else:
                # copy old token
                new_encoding.append(old_encoding[i])
                i += 1

        # count new pairs
        new_pair_counter = Counter()
        for i in range(len(new_encoding) - 1):
            new_pair_counter[(new_encoding[i], new_encoding[i+1])] += 1

        # transform counters to flat data structure, for easier comparison to CPython implementation
        old_pairs = [(a, b, k) for (a, b), k in old_pair_counter.items()]
        new_pairs = [(a, b, k) for (a, b), k in new_pair_counter.items()]

        return old_pairs, new_encoding, new_pairs