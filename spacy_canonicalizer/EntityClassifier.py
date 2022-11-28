from functools import reduce
from itertools import groupby

from Levenshtein import distance
import numpy as np
from numpy.linalg import norm
import spacy

class EntityClassifier:
    def __init__(self):
        pass

    # Cosine Similarity :)
    def similarity(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def _select_contextually_relevant(self, entities, prior_total, nlp, num=None, cutoff_percentage=0.8):
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Returns the KB IDs for each entity in each doc, including NIL if there is
        no prediction.
                                                                                                                                                                                                    
        Adapted from https://spacy.io/api/entitylinker#predict
        """

        # sims = np.asarray([nlp(ent.span.doc.text)[ent.span.start:ent.span.end].similarity(nlp(ent.description)) if ent.description else 0 for ent in entities])
        sims = np.asarray([nlp(ent.span.doc.text).similarity(nlp(ent.description)) if ent.description else 0 for ent in entities])
        # sims = np.asarray([self.similarity(ent.span.doc._.trf_data.tensors[1][0], nlp(ent.description)._.trf_data.tensors[1][0]) if ent.description else 0 for ent in entities])
        prior_probs = np.asarray([ent.prior if ent.description else 0 for ent in entities]) # / prior_total
        if sims.shape != prior_probs.shape:
            raise ValueError(Errors.E161)
        scores = (
            prior_probs + sims - (prior_probs * sims)
        )

        best_indices = scores.argsort()
        if num:
            return np.asarray(entities)[best_indices].tolist()[-num:]
        else:
            cutoff = cutoff_percentage * scores.sum()
            top_ents = []
            total = 0
            for idx in np.flip(best_indices):
                top_ents.append(entities[idx])
                total += scores[idx]
                if total > cutoff:
                    break
            return top_ents

    def _get_grouped_by_length(self, entities):
        sorted_by_len = sorted(entities, key=lambda entity: len(entity.get_span()), reverse=True)

        entities_by_length = {}
        for length, group in groupby(sorted_by_len, lambda entity: len(entity.get_span())):
            entities = list(group)
            entities_by_length[length] = entities

        return entities_by_length

    def _filter_max_length(self, entities):
        entities_by_length = self._get_grouped_by_length(entities)
        max_length = max(list(entities_by_length.keys()))

        return entities_by_length[max_length]

    def _filter_most_similar_alias(self, entities):
        similarities = np.array(
            [distance(entity.get_span().text.lower(), entity.get_original_alias().lower()) for entity in entities])

        min_indices = np.where(similarities == similarities.min())[0].tolist()

        return [entities[i] for i in min_indices]

    def _select_max_prior(self, entities, num=None, cutoff_percentage=0.8):
        # priors = [entity.get_prior() for entity in entities]
        # return entities[np.argmax(priors)]
        if num:
            return sorted(entities, key=lambda ent: ent.get_prior(), reverse=True)[:num]
        else:
            cutoff = cutoff_percentage * sum(map(lambda ent: ent.get_prior(), entities))
            top_ents = []
            total = 0
            for ent in sorted(entities, key=lambda ent: ent.get_prior(), reverse=True):
                top_ents.append(ent)
                total += ent.get_prior()
                if total > cutoff:
                    break
            return top_ents

    def _select_min_id(self, entities, num=None):
        # ids = [entity.get_id() for entity in entities]
        # return entities[np.argmin(ids)]
        if num:
            return sorted(entities, key=lambda ent: ent.get_id())[:num]
        else:
            return entities

    def _get_casing_difference(self, word1, original):
        difference = 0
        for w1, w2 in zip(word1, original):
            if w1 != w2:
                difference += 1

        return difference

    def _filter_most_similar(self, entities, casing_filter):

        if casing_filter:
            similarities = np.array(
                [self._get_casing_difference(entity.get_span().text, entity.get_original_alias()) for entity in entities])
            min_indices = np.where(similarities == similarities.min())[0].tolist()
            entities = [entities[i] for i in min_indices]

        similarities = np.array(
            [distance(entity.get_span().text.lower(), entity.get_original_alias().lower()) for entity in entities])
        min_indices = np.where(similarities == similarities.min())[0].tolist()
        return [entities[i] for i in min_indices]

    def _filter_expected_types(self, entities, expected_types):
        if not expected_types:
            return entities

        def type_filter(entity, expected_types):
            entity_types = entity.get_categories(max_depth=10)
            return any(ex_type["id"] in entity_types for ex_type in expected_types)

        return list(filter(lambda ent: type_filter(ent, expected_types), entities))

    def __call__(self, entities, nlp, entity_limit=None, expected_types=None, similarity_filter=False):
        filtered_by_length = self._filter_max_length(entities)
        filtered_by_similarity = self._filter_most_similar(filtered_by_length, similarity_filter)
        filtered_by_expected_type = self._filter_expected_types(filtered_by_similarity, expected_types)
        if nlp:
            return self._select_contextually_relevant(filtered_by_expected_type, sum(map(lambda ent: ent.prior, entities)), nlp, entity_limit)
        else:
            return self._select_max_prior(filtered_by_expected_type, entity_limit)
        # return self._select_min_id(filtered_by_popularity, entity_limit)
