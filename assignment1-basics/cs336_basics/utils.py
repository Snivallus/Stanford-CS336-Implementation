import os
from typing import BinaryIO, List
import random


class TeeStdout:
    """
    Helper class to tee stdout to multiple streams, e.g. stdout and a file
    """
    # Initialize with multiple streams
    def __init__(self, *streams):
        self.streams = streams

    # Write data to all streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    # Flush all streams
    def flush(self):
        for s in self.streams:
            s.flush()


def find_chunk_boundaries(
    file: BinaryIO,
    split_special_token: bytes = b"<|endoftext|>",
    desired_num_chunks: int = 64,
    min_chunk_size: int = 4 * 1024 * 1024, # 4 MB
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

    chunk_size = max(min_chunk_size, file_size // desired_num_chunks)

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4 * 1024  # Read ahead by 4k bytes at a time

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


def sample_documents(
    file_path: str,
    split_special_token: bytes = b"<|endoftext|>",
    num_samples: int = 10,
    desired_num_chunks: int = 100,
    min_doc_size: int = 1 * 1024 * 1024,  # 1 MB
    seed: int = 51
) -> List[str]:
    """
    Sample `num_samples` documents from a large file without reading it all into memory.

    Steps:
    1. Compute approximate chunk boundaries using `find_chunk_boundaries`.
    2. Randomly pick `num_samples` chunks.
    3. Read each chunk from file and return as UTF-8 string.

    Returns:
        List of sampled document strings.
    """
    sampled_docs = []

    with open(file_path, "rb") as f:
        # Compute chunk boundaries
        boundaries = find_chunk_boundaries(
            file = f,
            split_special_token = split_special_token,
            desired_num_chunks = desired_num_chunks,
            min_chunk_size = min_doc_size
        )

        # Each chunk = (start, end)
        chunks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

        # Sample chunks
        rng = random.Random(seed)
        sampled_chunks = rng.sample(chunks, min(num_samples, len(chunks)))

        # Read the sampled chunks
        for start, end in sampled_chunks:
            f.seek(start)
            chunk_bytes = f.read(end - start)

            # decode to string
            try:
                chunk_str = chunk_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # fallback: replace undecodable bytes
                chunk_str = chunk_bytes.decode("utf-8", errors="replace")

            sampled_docs.append(chunk_str)

    return sampled_docs