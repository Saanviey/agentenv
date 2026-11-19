def reverse_words(sentence):
    """Reverse the order of the words."""
    return " ".join(sentence.split(" ")[::1])
