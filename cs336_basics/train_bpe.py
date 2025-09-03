from collections import deque
from typing import Counter


class BPETokenizer:
    """Byte-Pair Encoding (BPE) Tokenizer implementation.
    
    This tokenizer implements the BPE algorithm for subword tokenization,
    similar to the GPT-2 tokenizer. It maintains vocabulary mappings and
    BPE merge operations to tokenize text into subword units.
    """
    
    def __init__(self):
        """Initialize the BPETokenizer with empty vocabularies and merge rules."""
        
        #: dict[int, str]: Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab: dict[int, str] = {}
        
        #: dict[str, int]: Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab: dict[str, int] = {}
        
        #: dict[tuple[int, int], int]: Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges: dict[tuple[int, int], int] = {}

        #: dict[tuple[str, str], int]: Dictionary of BPE merge ranks: 
        #: {(string_A, string_B): rank}, where lower rank = higher priority
        self.bpe_ranks: dict[tuple[str, str], int] = {}

    def train(self, input_path: str, vocab_size: int, speical_tokens: list[str]):
        """
        Train the BPE tokenizer from scratch.

        Args:
            input_path (str): Path to a text file with BPE tokenizer training data.
            vocab_size (int): A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary,vocabulary items produced from merging,and any special tokens).
            speical_tokens (list[str]): A list of strings to add to the vocabulary.The sespecial tokens do not otherwise affect BPE training.

        """
        # Preprocess: Replace spaces with "Ġ"
        # Note that Ġ is a particularity of the GPT-2 BPE implementation
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters, including "Ġ" if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted(set(processed_text)) if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if speical_tokens:
            for token in speical_tokens:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id


    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced