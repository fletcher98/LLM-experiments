'''\
This project implements a watermarking system for language model outputs.
A simple watermarking technique is employed where a secret subset of the vocabulary,
referred to as the 'green list', is randomly selected using a secret seed and a
defined gamma ratio. When generating text, the language model is biased towards
selecting tokens from this green list.

A statistical detection module then analyzes the output by applying a binomial test
to determine if the proportion of green tokens is significantly higher than expected,
thereby flagging potential watermarked content.

To further assess the robustness of the watermark, a paraphrasing module simulates
adversarial attacks by replacing words with synonyms, testing if the watermark signal
survives common text transformations. The resulting analysis aids in understanding
both the effectiveness and resilience of the watermarking scheme in real-world
scenarios.
'''
import random
import re
from scipy.stats import binom_test  # For performing the binomial test

# -------------------------------
# Watermarking Module
# -------------------------------
class Watermarker:
    """
    Implements a simple watermarking scheme where a secret subset (green list)
    is chosen from the vocabulary. When generating text, tokens belonging to
    the green list are favored.
    """
    def __init__(self, vocab, seed=42, gamma=0.5):
        """
        Parameters:
        - vocab: list of tokens (the vocabulary)
        - seed: random seed for reproducibility
        - gamma: fraction of tokens to designate as green (e.g., 0.5 means half the tokens)
        """
        self.vocab = vocab
        self.gamma = gamma
        self.seed = seed
        random.seed(seed)
        # Randomly sample a subset of the vocabulary to be the green list
        green_count = int(gamma * len(vocab))
        self.green_set = set(random.sample(vocab, green_count))
    
    def watermark_text(self, text):
        """
        Tokenizes text and counts the number of tokens that belong to the green list.
        Returns: (tokens, count of green tokens, total number of tokens)
        """
        # Simple tokenization: lower-case and split on word characters
        tokens = re.findall(r'\w+', text.lower())
        green_count = sum(1 for token in tokens if token in self.green_set)
        total = len(tokens)
        return tokens, green_count, total

# -------------------------------
# Detection Module
# -------------------------------
class Detector:
    """
    Uses statistical testing to detect if a given text was generated with the watermark.
    The hypothesis is that watermarked text will have a higher-than-expected proportion
    of green tokens.
    """
    def __init__(self, gamma):
        self.gamma = gamma  # Expected probability of a green token under null hypothesis
    
    def detect(self, text, watermarker):
        """
        Applies the watermark detector to text.
        Returns:
        - p_value: the p-value from the binomial test (lower suggests watermarked)
        - green_proportion: fraction of tokens that are green
        """
        tokens, green_count, total = watermarker.watermark_text(text)
        if total == 0:
            return 1.0, 0
        # Under the null hypothesis (non-watermarked), the probability of a green token is gamma.
        # We test if the observed green_count is significantly higher.
        p_value = binom_test(green_count, total, self.gamma, alternative='greater')
        return p_value, green_count / total

# -------------------------------
# Paraphrasing Module (Robustness Evaluation)
# -------------------------------
class Paraphraser:
    """
    A simple paraphraser that replaces certain words with synonyms.
    In a real-world scenario, I would use a more sophisticated model.
    """
    def __init__(self, synonym_dict=None):
        # Define a basic synonym dictionary if none provided.
        if synonym_dict is None:
            self.synonym_dict = {
                'good': ['great', 'excellent', 'favorable'],
                'bad': ['poor', 'inferior', 'substandard'],
                'happy': ['joyful', 'cheerful', 'content'],
                'sad': ['unhappy', 'sorrowful', 'dejected'],
                # Additional words and synonyms can be added here.
            }
        else:
            self.synonym_dict = synonym_dict
    
    def paraphrase_text(self, text, replacement_prob=0.3):
        """
        Paraphrases the input text by randomly replacing words (if found in the synonym dictionary)
        with one of their synonyms.
        """
        words = text.split()
        new_words = []
        for word in words:
            # Strip punctuation for matching (a more advanced approach might preserve punctuation)
            key = word.lower().strip('.,!?')
            if key in self.synonym_dict and random.random() < replacement_prob:
                synonym = random.choice(self.synonym_dict[key])
                # Preserve capitalization if necessary
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym)
            else:
                new_words.append(word)
        return ' '.join(new_words)

# -------------------------------
# Evaluation Module
# -------------------------------
def evaluate_watermarking():
    # Example vocabulary – in practice, this would be the LLM’s full vocabulary.
    vocab = ['the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'for', 'it',
             'with', 'as', 'on', 'at', 'by', 'this', 'be', 'or', 'are', 'from',
             'good', 'bad', 'happy', 'sad', 'quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
    
    # Initialize the watermarking module with a chosen gamma (e.g., 0.5 means 50% green tokens)
    watermarker = Watermarker(vocab, seed=123, gamma=0.5)
    
    # Simulated LLM output (this text would be generated by a watermarked LLM)
    generated_text = (
        "The quick brown fox jumps over the lazy dog. "
        "It is a good day and everyone is happy with the outcomes."
    )
    
    # Detect watermark on the original text
    detector = Detector(gamma=0.5)
    p_value, proportion = detector.detect(generated_text, watermarker)
    print("Original text detection:")
    print("P-value:", p_value, "Green token proportion:", proportion)
    
    # Paraphrase the text to simulate an adversarial attack
    paraphraser = Paraphraser()
    paraphrased_text = paraphraser.paraphrase_text(generated_text, replacement_prob=0.5)
    
    # Detect watermark on the paraphrased text
    p_value_para, proportion_para = detector.detect(paraphrased_text, watermarker)
    print("\nParaphrased text detection:")
    print("Paraphrased text:", paraphrased_text)
    print("P-value:", p_value_para, "Green token proportion:", proportion_para)

if __name__ == '__main__':
    evaluate_watermarking()