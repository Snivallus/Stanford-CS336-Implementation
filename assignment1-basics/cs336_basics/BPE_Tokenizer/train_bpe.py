"""
Byte Pair Encoding (BPE) training implementation.

This module trains a BPE vocabulary from raw text by:
- Pretokenizing input with a GPT-2–style regex
- Counting words in parallel using memory-mapped I/O
- Iteratively merging the most frequent byte/token pairs using a lazy max-heap approach
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

WordEncoding = List[TokenId]
Vocab = Dict[TokenId, bytes]
WordCounter = Counter[Word]
PairCounter = Dict[Pair, int]
PairStrings = Dict[Pair, Tuple[bytes, bytes]]
PairToWords = DefaultDict[Pair, Counter[Word]]

import cs336_basics.BPE_Tokenizer.max_heapq as maxheap


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
        # unpack args
        input_path, start, end, gpt2_pattern, special_token_pattern = args

        # split by special tokens and count in each block
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
        pair_counts = defaultdict(int) # hash map from pair to its occurrence count in the corpus 
        # count pairs in all words
        for word, count in word_counts.items():
            encoding = word_encodings[word]
            for i in range(0, len(encoding) - 1):
                pair = encoding[i], encoding[i+1]
                # update corpus-level pair count (with word count)
                pair_counts[pair] += count
                # update word-level structural count
                pair_to_words[pair][word] += 1
                # record pair strings if not recorded
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
        # get the most frequent pair
        while pair_heap:
            # pop max
            count, string_priority, pair_to_be_merged = maxheap.heappop_max(pair_heap)
            # check pair validity
            if pair_to_be_merged in pair_counts:
                if pair_counts[pair_to_be_merged] == count:
                    # valid pair
                    break
                else:
                    # outdated count, push updated count
                    maxheap.heappush_max(
                        pair_heap, 
                        (pair_counts[pair_to_be_merged], string_priority, pair_to_be_merged)
                    )
        else: # no break => merge_pair not found (should not happen)
            Warning("no valid pairs found!")
            return False

        # perform merge
        merged_bytes = vocab[pair_to_be_merged[0]] + vocab[pair_to_be_merged[1]]
        # add new token to vocabulary
        vocab[size] = merged_bytes
        # new token id
        new_id = size

        # get affected words
        affected_words = pair_to_words[pair_to_be_merged].copy() # copy to avoid modification during iteration
        # update counts of affected words
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

        # record the merge
        merges.append((vocab[pair_to_be_merged[0]], vocab[pair_to_be_merged[1]]))


    @staticmethod
    def _update_pair_count_of_affected_words(
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
        pair_count_difference = defaultdict(int) # hash map from pair to count difference
        bytes_a, bytes_b = pair_to_be_merged
        for word in affected_words:
            # get affected tokens and word counts
            old_encoding = word_encodings[word]
            word_freq = word_counts[word]

            # Count old pairs in this word
            old_pair_counter = Counter()
            for i in range(len(old_encoding) - 1):
                old_pair_counter[(old_encoding[i], old_encoding[i+1])] += 1

            # Merge pair_to_be_merged
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

            word_encodings[word] = new_encoding

            # Count new pairs
            new_pair_counter = Counter()
            for i in range(len(new_encoding) - 1):
                new_pair_counter[(new_encoding[i], new_encoding[i+1])] += 1

            # Remove old pair contributions
            for pair, k in old_pair_counter.items():
                if k == 0:
                    continue

                # update pair_to_words
                cnt = pair_to_words[pair][word] - k
                if cnt > 0:
                    pair_to_words[pair][word] = cnt
                else:
                    del pair_to_words[pair][word]

                # update global diff
                pair_count_difference[pair] -= k * word_freq

            # Add new pair contributions
            for pair, k in new_pair_counter.items():
                if k == 0:
                    continue

                pair_to_words[pair][word] += k
                pair_count_difference[pair] += k * word_freq

        # Apply global pair count differences
        for pair, diff in pair_count_difference.items():
            if diff == 0:
                continue

            new_count = pair_counts.get(pair, 0) + diff

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