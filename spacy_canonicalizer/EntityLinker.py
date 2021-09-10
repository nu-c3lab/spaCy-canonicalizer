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

    def __call__(self, doc, **component_cfg):
        tce = TermCandidateExtractor(doc, component_cfg.get("single_term", False))
        classifier = EntityClassifier()

        for sent in doc.sents:
            sent._.linkedEntities = EntityCollection([])

        entities = []
        for termCandidates in tce:
            entityCandidates = termCandidates.get_entity_candidates()
            if len(entityCandidates) > 0:
                entity = classifier(entityCandidates, component_cfg.get("expected_types", None))
                if entity:
                    entity.span.sent._.linkedEntities.append(entity)
                    entities.append(entity)

        doc._.linkedEntities = EntityCollection(entities)

        return doc
