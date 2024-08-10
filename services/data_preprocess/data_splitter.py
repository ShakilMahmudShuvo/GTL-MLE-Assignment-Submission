from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, test_size=0.1, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, texts, labels):
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=self.test_size, random_state=self.random_state)
        return train_texts, val_texts, train_labels, val_labels
