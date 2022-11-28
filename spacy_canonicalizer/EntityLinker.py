from functools import reduce

import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language

from .EntityClassifier import EntityClassifier
from .EntityCollection import EntityCollection
from .TermCandidateExtractor import TermCandidateExtractor

@Language.factory('entityLinker')
class EntityLinker:

    def __init__(self, nlp, name):
        Doc.set_extension("linkedEntities", default=EntityCollection(), force=True)
        Span.set_extension("linkedEntities", default=None, force=True)
        Doc.set_extension("unlinkedEntities", default=[], force=True)
        Span.set_extension("unlinkedEntities", default=None, force=True)

    def __call__(self, doc, **component_cfg):
        nlp = component_cfg.get("nlp", None)
        tce = TermCandidateExtractor(doc, component_cfg.get("stanza_nlp", None), component_cfg.get("single_term", False))
        classifier = EntityClassifier()

        for sent in doc.sents:
            sent._.linkedEntities = EntityCollection([])
            sent._.unlinkedEntities = []

        entities = []
        for termCandidates in tce:
            entityCandidates = termCandidates.get_entity_candidates()
            if len(entityCandidates) > 0:
                for entity in classifier(entityCandidates, nlp, component_cfg.get("entity_limit", None), component_cfg.get("expected_types", None), component_cfg.get("similarity_filter", False)):
                    if entity.get_id() not in map(lambda ent: ent.get_id(), entity.span.sent._.linkedEntities.entities):
                        entity.span.sent._.linkedEntities.append(entity)
                    if entity.get_id() not in map(lambda ent: ent.get_id(), entities):
                        entities.append(entity)
            unlinkedEntity = max(termCandidates.variations, key=len)
            if unlinkedEntity not in map(lambda linkedEntity: linkedEntity.span, unlinkedEntity.sent._.linkedEntities):
                unlinkedEntity.sent._.unlinkedEntities.append(unlinkedEntity)

        doc._.linkedEntities = EntityCollection(entities)
        doc._.unlinkedEntities = reduce(lambda a, sent: a + sent._.unlinkedEntities, list(doc.sents), []) # TODO: Consider doing the same for doc._.linkedEntities for simplicity

        return doc
