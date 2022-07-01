from itertools import groupby
import numpy as np


class EntityClassifier:
    def __init__(self):
        pass

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

    def _select_max_prior(self, entities):
        priors = [entity.get_prior() for entity in entities]
        return entities[np.argmax(priors)]

    def _select_min_id(self, entities, num):
        # ids = [entity.get_id() for entity in entities]
        # return entities[np.argmin(ids)]
        best_entities = []
        for i in range(min(num, len(entities))):
            best_ent = min(entities, key=lambda ent: ent.get_id())
            best_entities.append(best_ent)
            entities.remove(best_ent)
        return best_entities

    def _get_casing_difference(self, word1, original):
        difference = 0
        for w1, w2 in zip(word1, original):
            if w1 != w2:
                difference += 1

        return difference

    def _filter_most_similar(self, entities):
        similarities = np.array(
            [self._get_casing_difference(entity.get_span().text, entity.get_original_alias()) for entity in entities])

        min_indices = np.where(similarities == similarities.min())[0].tolist()

        return [entities[i] for i in min_indices]

    def _filter_expected_types(self, entities, expected_types):
        if not expected_types:
            return entities

        def type_filter(entity, expected_types):
            entity_types = entity.get_categories(max_depth=10)
            return any(ex_type["id"] in entity_types for ex_type in expected_types)

        return list(filter(lambda ent: type_filter(ent, expected_types), entities))

    def __call__(self, entities, expected_types=None):
        filtered_by_length = self._filter_max_length(entities)
        filtered_by_casing = self._filter_most_similar(filtered_by_length)
        filtered_by_expected_type = self._filter_expected_types(filtered_by_casing, expected_types)

        return self._select_min_id(filtered_by_expected_type, 2) if len(filtered_by_expected_type) else [] # Instead of using a fixed number here, probably want to use some sort of proximity metric
