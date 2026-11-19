class WordCounter:
    def __init__(self):
        self.counts = {}

    def add(self, word):
        word = word.lower()
        self.counts[word] = 1

    def most_common(self):
        return max(self.counts, key=self.counts.get)
