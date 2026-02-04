import os
import mmap
import gc
import pickle
import json
import regex as re
import ast
import multiprocessing as mp
from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Optional,
    Iterable, 
    Iterator
)

from cs336_basics.utils import find_chunk_boundaries


N_BYTES = 256
MAX_NUM_PROCESSES = 64
MIN_CHUNK_SIZE = 256 * 1024 # 256 KB

Word = bytes
TokenId = int
Vocab = Dict[TokenId, bytes]
Byte2ID = Dict[bytes, TokenId]
Merge = Tuple[bytes, bytes]
MergeRanks = Dict[Merge, int]


class BPE_Tokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer.
    """
    def __init__(
        self,
        vocab: Vocab,
        merges: List[Merge],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize the tokenizer.

        Parameters:
        - vocab (Vocab): Dictionary mapping token IDs to their byte string representation.
        - merges (List[Merge]): List of byte pair merges applied during training.
        - special_tokens (Optional[List[str]]): Optional list of special tokens.
        """
        
        # --- Vocab ---
        self.vocab: Vocab = vocab
        self.byte_to_id: Byte2ID = {v: k for k, v in vocab.items()} # reverse vocab
        
        # -- Merges ---
        self.merges: List[Merge] = merges
        self.merge_ranks: MergeRanks = {
            merge: i for i, merge in enumerate(merges) # Merge ranks (lower index = higher priority)
        }

        # --- Special tokens ---
        self.special_tokens = special_tokens
        self.special_token_pattern = None
        self.special_token_bytes = {}
        
        if self.special_tokens is not None:
            self.special_tokens = sorted(self.special_tokens, key = len, reverse = True)
            self.special_token_pattern = re.compile(
                b"|".join(
                    re.escape(token.encode("utf-8")) 
                    for token in self.special_tokens
                )
            )

            # append special tokens into vocab if missing
            next_id = max(self.vocab.keys(), default=-1) + 1

            for token in self.special_tokens:
                b = token.encode("utf-8")

                if b not in self.byte_to_id:
                    self.vocab[next_id] = b
                    self.byte_to_id[b] = next_id
                    next_id += 1

                self.special_token_bytes[b] = self.byte_to_id[b]

        # --- GPT-2 regex pattern ---
        self.gpt2_pattern = re.compile(
            rb"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"""
        )


    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPE_Tokenizer":
        """
        Construct and return a BPE_Tokenizer from serialized vocab + merges.

        Supported formats:
        - vocab: .pkl / .pickle / .json
        - merges: .pkl / .pickle / .txt

        vocab json format should be:
            { "<token_str>" : <token_id>, ... }
        where token_str is either a valid UTF-8 string or a printable escape format
        like "\\x00\\xff".

        merges txt format should be:
            each line: "<token1> <token2>"
        where tokens are interpreted similarly to vocab json keys.
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab file not found: {vocab_path}")

        if not os.path.exists(merges_path):
            raise FileNotFoundError(f"merges file not found: {merges_path}")

        def parse_token_str(token_str: str) -> bytes:
            """
            Convert token string representation back into bytes.

            Supports:
            - normal UTF-8 strings, e.g. "hello"
            - hex escaped sequences like "\\x00\\x1f"
            """
            if "\\x" in token_str:
                # interpret \xHH escapes
                out = bytearray()
                i = 0
                while i < len(token_str):
                    if token_str[i] == "\\" and i + 3 < len(token_str) and token_str[i + 1] == "x":
                        hex_part = token_str[i + 2 : i + 4]
                        out.append(int(hex_part, 16))
                        i += 4
                    else:
                        # normal unicode char
                        out.extend(token_str[i].encode("utf-8"))
                        i += 1
                return bytes(out)

            return token_str.encode("utf-8")

        # -----------------------------
        # Load vocab
        # -----------------------------
        if vocab_path.endswith((".pkl", ".pickle")):
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)

        elif vocab_path.endswith(".json"):
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw_vocab = json.load(f)

            # raw_vocab is {token_str: token_id}
            vocab = {}
            for token_str, token_id in raw_vocab.items():
                if not isinstance(token_id, int):
                    raise ValueError(f"Invalid vocab id type: {type(token_id)} for token {token_str!r}")

                token_bytes = parse_token_str(token_str)
                vocab[token_id] = token_bytes

        else:
            raise ValueError(
                f"Unsupported vocab file format: {vocab_path}. "
                "Expected .pkl/.pickle/.json"
            )

        # -----------------------------
        # Load merges
        # -----------------------------
        if merges_path.endswith((".pkl", ".pickle")):
            with open(merges_path, "rb") as f:
                merges = pickle.load(f)

        elif merges_path.endswith(".txt"):
            merges = []
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    if line.lstrip().startswith("#"):
                        continue

                    try:
                        # split line (e.g. "b' Snap' b'e '") into token1 (e.g. "b' Snap'") and token2 (e.g. "b'e ')
                        parts = line.split(" ", 1)
                        if len(parts) != 2:
                            raise ValueError(f"Invalid merge line (no space found): {line!r}")
                        a_bytes = ast.literal_eval(parts[0])
                        b_bytes = ast.literal_eval(parts[1])
                        if not isinstance(a_bytes, bytes) or not isinstance(b_bytes, bytes):
                            raise ValueError(f"Invalid merge line (not bytes): {line!r}")
                        merges.append((a_bytes, b_bytes))
                    except Exception:
                        # split line (e.g. " Snap e ") into token1 (eg. " Snap") and token2 (e.g. "e ")
                        # find the first space that is between two non-space characters
                        line_stripped = line.rstrip("\n")
                        split_idx = None
                        for i, c in enumerate(line_stripped):
                            if c == " ":
                                left = line_stripped[:i]
                                right = line_stripped[i + 1 :]
                                if left.strip() != "" or right.strip() != "":
                                    split_idx = i
                                    break

                        if split_idx is None:
                            raise ValueError(f"Invalid merge line (cannot split): {line_stripped!r}")

                        token1_str = line_stripped[:split_idx]
                        token2_str = line_stripped[split_idx + 1 :]

                        # convert to bytes (support \xHH escaped sequences if needed)
                        merges.append((parse_token_str(token1_str), parse_token_str(token2_str)))

        else:
            raise ValueError(
                f"Unsupported merges file format: {merges_path}. "
                "Expected .pkl/.pickle/.txt"
            )

        # -----------------------------
        # Construct tokenizer
        # -----------------------------
        return cls(
            vocab = vocab,
            merges = merges,
            special_tokens = special_tokens,
        )


    def decode(
        self, 
        token_ids: List[TokenId]
    ) -> str:
        """
        Decode a list of token IDs into a string.
        """
        byte_stream = b"".join(self.vocab[i] for i in token_ids)
        return byte_stream.decode("utf-8", errors = "replace")


    def encode(
        self, 
        input: Union[bytes, str],
        max_num_processes: int = MAX_NUM_PROCESSES,
    ) -> List[TokenId]:
        """
        Encode a string or bytes or a large text file into a list of token IDs.
        """
        encoded: List[TokenId] = []

        if isinstance(input, str):
            if not os.path.exists(input):
                input = input.encode("utf-8")
            else:
                input_path = input
                # find chunk boundaries
                with open(input_path, "rb") as f:           
                    boundaries = find_chunk_boundaries(
                        file = f,
                        split_special_token = b"<|endoftext|>",
                        desired_num_chunks = max_num_processes,
                        min_chunk_size = MIN_CHUNK_SIZE
                    )

                # create chunk ranges
                ranges = [
                    (
                        i, # <- chunk index
                        input_path,
                        boundaries[i],
                        boundaries[i + 1],
                        self.gpt2_pattern,
                        self.special_token_pattern,
                        self.special_token_bytes,
                        self.merge_ranks,
                        self.byte_to_id,
                    )
                    for i in range(len(boundaries) - 1)
                ]

                # leave half CPUs free for OS and I/O
                num_procs = min(
                    max(1, os.cpu_count() // 2),
                    len(ranges),
                )
                
                # preallocate result slots
                chunk_results = [None] * len(ranges)

                with mp.Pool(processes = num_procs) as pool:
                    for idx, ids in pool.imap_unordered(
                        self._encode_chunk_worker,
                        ranges,
                        chunksize = 1,
                    ):
                        chunk_results[idx] = ids

                # --- stitch chunks in order ---
                encoded = []
                for ids in chunk_results:
                    encoded.extend(ids)

                gc.collect() # force garbage collection to free memory

                return encoded

        if isinstance(input, (bytes, bytearray)):
            data = bytes(input)

            if self.special_tokens is None or len(self.special_tokens) == 0:
                blocks = [data]
                specials = []
            else:
                blocks = self.special_token_pattern.split(data)
                specials = self.special_token_pattern.findall(data)

            for i, block in enumerate(blocks):
                for m in self.gpt2_pattern.finditer(block):
                    encoded.extend(
                        _encode_word(
                            m.group(0),
                            self.byte_to_id,
                            self.merge_ranks,
                        )
                    )

                if i < len(specials):
                    encoded.append(self.special_token_bytes[specials[i]])

            return encoded
        
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")


    @staticmethod
    def _encode_chunk_worker(args):
        (
            idx, # <- chunk index
            input_path, start, end,
            gpt2_pattern,
            special_token_pattern,
            special_token_bytes,
            merge_ranks,
            byte_to_id
        ) = args

        with open(input_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            data = mm[start:end]

            token_ids = []

            if len(special_token_bytes) == 0:
                blocks = [data]
                specials = []
            else:
                blocks = special_token_pattern.split(data)
                specials = special_token_pattern.findall(data)

            for i, block in enumerate(blocks):
                for m in gpt2_pattern.finditer(block):
                    token_ids.extend(
                        _encode_word(
                            m.group(0),
                            byte_to_id,
                            merge_ranks
                        )
                    )

                if i < len(specials):
                    token_ids.append(special_token_bytes[specials[i]])

            del data
            mm.close()

        return idx, token_ids


    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[TokenId]:
        """
        Lazily encode an iterable of strings (e.g., a file handle).

        This is memory-efficient because:
        - We do not load the whole file into RAM
        - We do not store the full token list
        - We yield token IDs one-by-one

        Example usage:
            with open("data.txt", "r", encoding="utf-8") as f:
                for tid in tokenizer.encode_iterable(f):
                    ...

        Notes:
        - Each element of `iterable` is assumed to be a string (usually a line).
        - Special tokens are handled correctly (no merging across special token boundaries).
        """
        for text in iterable:
            if not isinstance(text, str):
                raise TypeError(
                    f"encode_iterable expects Iterable[str], but got element type {type(text)}"
                )

            # convert to bytes (UTF-8)
            data = text.encode("utf-8")

            # split by special tokens if enabled
            if self.special_tokens is None or len(self.special_tokens) == 0:
                blocks = [data]
                specials = []
            else:
                blocks = self.special_token_pattern.split(data)
                specials = self.special_token_pattern.findall(data)

            for i, block in enumerate(blocks):
                # GPT-2 regex pre-tokenization
                for m in self.gpt2_pattern.finditer(block):
                    word = m.group(0)

                    # BPE encode one word -> token ids
                    for tid in _encode_word(word, self.byte_to_id, self.merge_ranks):
                        yield tid

                # emit special token if present
                if i < len(specials):
                    yield self.special_token_bytes[specials[i]]


def _encode_word(
    word: Word, 
    byte_to_id: Byte2ID,
    merge_ranks: MergeRanks, 
)-> List[TokenId]:
    symbols = [bytes([b]) for b in word]

    while True:
        pairs = [
            (symbols[i], symbols[i + 1])
            for i in range(len(symbols) - 1)
        ]

        ranked = [
            (merge_ranks[p], i, p)
            for i, p in enumerate(pairs)
            if p in merge_ranks
        ]

        if not ranked:
            break

        _, idx, (a, b) = min(ranked)
        symbols[idx:idx + 2] = [a + b]

    return [byte_to_id[s] for s in symbols]