import regex as re
from collections import defaultdict, Counter
import multiprocessing as mp
import time
import argparse
import cs336_basics.BPE_Tokenizer.max_heapq as maxheap

CHUNK_SIZE = 1024 * 1024
N_BYTES = 256
NUM_COUNTER_PROCESS = 16

class BPE_Trainer():
    def train_bpe(
        self, 
        input_path, 
        vocab_size, 
        special_tokens, 
        *args
    ):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num_counter", 
            "-c",                
            type=int, 
            default=NUM_COUNTER_PROCESS, 
            help="number of processes for counting"
        )
        parser.add_argument(
            "--do_monitor",
            action="store_true",
            help="Enable queue monitor. (default: False)"
        )        

        args = parser.parse_args(args)
        print(f"train_bpe: {args=}")
        num_counter = args.num_counter
        do_monitor = args.do_monitor 

        # pretokenize and count words
        start_time = time.perf_counter()
        word_counts = self._pretokenize_and_count_mp(
            input_path, 
            special_tokens, 
            num_counter, 
            do_monitor
        )
        end_time = time.perf_counter()
        print(f"_pretokenize_and_count_mp: {end_time - start_time}")
        
        # initialize vocabulary
        vocab = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            vocab[N_BYTES + i] = token.encode('utf-8')
        size = N_BYTES + len(special_tokens)
        merges = []

        # encode words to byte ids
        word_encodings = {} # hash map from word to list of byte ids
        for word in word_counts:
            word_encodings[word] = list(word.encode('utf-8'))

        # count initial pairs
        pair_strings = {} # hash map from pair to its string representation (tuple of strings)
        pair_to_words = defaultdict(Counter) # hash map from pair to words containing the pair (with occurrence count of the pair in each word)
        start_time = time.perf_counter()
        pair_counts = BPE_Trainer._count_pairs(
            word_counts, 
            word_encodings, 
            pair_strings, 
            vocab, 
            pair_to_words
        )
        end_time = time.perf_counter()
        print(f"_count_pairs: {end_time - start_time:.2f}s")

        # build maxheap
        start_time = time.perf_counter()
        pair_heap = []
        for pair, count in pair_counts.items():
            maxheap.heappush_max(
                pair_heap, 
                (count, pair_strings[pair], pair)
            )
        end_time = time.perf_counter()
        print(f"build heap: {end_time - start_time:.2f}s")

        # perform merges
        start_time = time.perf_counter()
        while size < vocab_size:
            BPE_Trainer._merge_a_pair(
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
        print(f"merge time: {end_time - start_time}")               
        
        return vocab, merges


    def _pretokenize_and_count_mp(
            self,
            input_path: str, 
            special_tokens: list[str],
            num_counter, 
            do_monitor
        ):
        # GPT-2 regex
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # build split pattern
        special_token_pattern = "|".join(re.escape(token) for token in special_tokens)

        chunk_queue = mp.Queue(maxsize=1_000_000)
        counter_queue = mp.Queue(maxsize=1_000_000)
        counter_processes = []
     
        # start counter processes
        for i in range(num_counter):
            p = mp.Process(
                target = BPE_Trainer._chunk_counter_process, 
                args = (
                    chunk_queue, 
                    counter_queue, 
                    PAT, 
                    special_token_pattern
                ),
                name = f"counter_process-{i+1}")
            p.start()
            counter_processes.append(p)    

        # For unit tests, call stop_event.set() to terminate the monitor early.
        # Otherwise, the monitor process may cause the speed test to fail.
        if do_monitor:
            stop_event = mp.Event()
            monitor_process = mp.Process(
                target = BPE_Trainer._queue_moniter_process, 
                args=(
                    chunk_queue, 
                    counter_queue, 
                    stop_event
                )
            )
            monitor_process.start()

        # feed chunks to chunk_queue
        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            chunk_queue.put(chunk)

        # after all chunks are fed, signal counter processes to stop
        for i in range(num_counter):
            chunk_queue.put(None)

        # counter processes finished already, now drain counter_queue
        word_counts = Counter()
        finished = 0
        while finished < num_counter:
            counter = counter_queue.get()
            if counter is None:
                finished += 1
            else:
                word_counts.update(counter)

        # wait for all counter processes to finish
        for p in counter_processes:
            p.join()

        # after all counters are finished, stop monitor process
        if do_monitor:
            stop_event.set()
            monitor_process.join() 

        return word_counts


    @staticmethod
    def _chunk_counter_process(
        chunk_queue, 
        counter_queue, 
        pattern, 
        special_token_pattern
    ):
        while True:
            # get chunk
            chunk = chunk_queue.get()
            if chunk == None:
                break
            # split by special tokens
            blocks = re.split(special_token_pattern, chunk)
            # count tokens in each block
            counter = defaultdict(int)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    counter[text] += 1
            # put counter to counter_queue
            counter_queue.put(counter)
        
        # signal main process this worker is done
        counter_queue.put(None)


    @staticmethod
    def _queue_moniter_process(
        chunk_queue, 
        counter_queue,
        event
    ):
        # monitor queue sizes every 10 seconds
        while not event.is_set():
            print(f"""
                chunk_queue: {chunk_queue.qsize()}, 
                counter_queue: {counter_queue.qsize()}
            """)
            time.sleep(10)


    @staticmethod
    def _chunk_documents_streaming(
        path: str,
        chunk_size: int = CHUNK_SIZE,
        special_token: str = "<|endoftext|>"
    ):
        """
        Reads 'path' in streaming fashion, yielding chunks of text that
        each end on a '<|endoftext|>' boundary.
        """

        leftover = ""
        token_len = len(special_token)

        with open(path, "r", encoding="utf-8") as f:
            while True:
                # read ahead chunk_size bytes, end if EOF
                block = f.read(chunk_size)
                if not block:
                    break

                # prepend leftover from last read
                block = leftover + block
                leftover = ""

                # find last <|endoftext|> in block
                last_eot_idx = block.rfind(special_token)

                # if no <|endoftext|> found, keep reading
                if last_eot_idx == -1:
                    leftover = block
                else:
                    # yield the block up to that boundary
                    yield block[: last_eot_idx + token_len]
                    # store leftover text
                    leftover = block[last_eot_idx + token_len:]

        # yield leftover if any
        if leftover:
            yield leftover


    @staticmethod    
    def _count_pairs(
        word_counts, 
        word_encodings, 
        pair_strings, 
        vocab, 
        pair_to_words
    ):
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


    @staticmethod
    def _merge_a_pair(
        pair_counts, 
        pair_strings, 
        vocab, 
        pair_to_words, 
        word_counts, 
        word_encodings, 
        merges, 
        size, 
        pair_heap
    ):
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
        BPE_Trainer._update_pair_count_of_affected_words(
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
    ):
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