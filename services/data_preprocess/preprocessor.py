from bnunicodenormalizer import Normalizer


class Preprocessor:
    def __init__(self):
        self.normalizer = Normalizer()

    def normalize_tokens(self, df):
        df['token'] = df['token'].apply(lambda x: self.normalizer(x)['normalized'])
        return df

    def create_mappings(self, df):
        pos_tag_to_id = {tag: idx for idx, tag in enumerate(sorted(set(df['pos_tag'])))}
        ner_tag_to_id = {tag: idx for idx, tag in enumerate(sorted(set(df['ner_tag'])))}
        return pos_tag_to_id, ner_tag_to_id

    def prepare_dataset(self, df, pos_tag_to_id, ner_tag_to_id):
        grouped = df.groupby('sentence').agg(list)
        texts = grouped['token'].tolist()
        labels = [{'pos_tag': [pos_tag_to_id[tag] for tag in pos_tags], 'ner_tag': [ner_tag_to_id[tag] for tag in
                                                                                    ner_tags]} for pos_tags, ner_tags in
                  zip(grouped['pos_tag'], grouped['ner_tag'])]
        return texts, labels
