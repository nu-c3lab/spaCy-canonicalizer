from functools import reduce

from .TermCandidate import TermCandidate

from nltk import Tree

class TermCandidateExtractor:
    def __init__(self, doc, stanza_nlp, single_term):
        self.doc = doc
        self.single_term = single_term
        self.punctuation = ['-', ',', '.', '!', '?']
        # These come from here: http://surdeanu.cs.arizona.edu//mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        self.word_noun_pos = ['CD', 'FW', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'SYM'] #, 'WP', 'WP$']
        self.extended_word_noun_pos = ['JJ', 'JJR', 'JJS', 'PDT', 'VBG', 'VBN', '-LRB-', '-RRB-'] #, 'WDT']
        self.phrase_noun_pos = ['NP'] #, 'WHNP']
        self.stanza_nlp = stanza_nlp

    def __iter__(self):
        for sent in self.doc.sents:
            for candidate in (self._get_candidates_from_constituency_tree(sent) if self.stanza_nlp else self._get_candidates_in_sent(sent, self.doc)):
                yield candidate

    # def _get_NPs(self, tree):
    #     return map(lambda subtree: subtree.leaves(), tree.subtrees(lambda subtree: subtree.label() in ['NP', 'NN']))

    def _add_charspans(self, tree, position, offsets):
        if len(position) == 1:
            tree[position[0]] = (tree[position[0]], offsets)
        else:
            return self._add_charspans(tree[position[0]], position[1:], offsets)


    def _get_candidates_from_constituency_tree(self, sent):

        def get_span(tree):
            try:
                leaf_parents = list(tree.subtrees(lambda t: t.label() in self.word_noun_pos + self.extended_word_noun_pos + ['DT']))
                if leaf_parents[0].label() == 'DT' and len(leaf_parents) > 2:
                    leaf_parents_no_det = leaf_parents[1:]
                    return [(min(map(lambda lp: lp[0][1][0], leaf_parents)), max(map(lambda lp: lp[0][1][1], leaf_parents))), (min(map(lambda lp: lp[0][1][0], leaf_parents_no_det)), max(map(lambda lp: lp[0][1][1], leaf_parents_no_det)))]
                else:
                    return [(min(map(lambda lp: lp[0][1][0], leaf_parents)), max(map(lambda lp: lp[0][1][1], leaf_parents)))]
            except (ValueError, IndexError):
                return [(min(map(lambda leaf: leaf [1][0], tree.leaves())), max(map(lambda leaf: leaf [1][1], tree.leaves())))]

        def get_candidates(tree, doc, term_candidates=None):

            for child in tree:
                if type(child) == Tree:
                    if child.label() in self.word_noun_pos:
                        doc_cand = doc.char_span(child[0][1][0],child[0][1][1])
                        # if doc_cand:
                        if term_candidates:
                            term_candidates.append(doc_cand)
                        else:
                            cands = TermCandidate(doc_cand)
                            candidates.append(cands)
                    elif child.label() in self.phrase_noun_pos and len(child) > 1:
                        doc_cands = [doc.char_span(*span) for span in get_span(child)]
                        # if doc_cand: # This check is necessary because the StanfordCoreNLP parser can potentially extract a sub-word token,
                                     # which will not correspond to an actual token in the spacy tokenization
                        if term_candidates:
                            # term_candidates.append(doc_cand)
                            term_candidates.extend(doc_cands)
                            get_candidates(child, doc, term_candidates)
                        else:
                            cands = TermCandidate(doc_cands)
                            get_candidates(child, doc, cands)
                            candidates.append(cands)
                    elif child.label() == 'PP': # New candidate set for prepositional phrase
                        get_candidates(child, doc, None)
                    else:
                        get_candidates(child, doc, term_candidates)

        candidates = []

        # parsed_data = parser.api_call(sent.text, properties={'ssplit.eolonly': 'true', 'tokenize.whitespace': 'false'})
        # parsed_sent = parsed_data["sentences"][0]

        doc = self.stanza_nlp(sent.text)
        tree = Tree.fromstring(str(doc.sentences[0].constituency))
        # tree.pretty_print() # TODO: remove when done testing

        for token in doc.sentences[0].tokens:
            self._add_charspans(tree, tree.leaf_treeposition(token.id[0]-1), (token.start_char, token.end_char))

        get_candidates(tree, sent)

        return candidates
        

    def _get_candidates_in_sent(self, sent, doc):

        root = list(filter(lambda token: token.dep_ == "ROOT", sent))[0]

        excluded_children = []
        candidates = []

        def get_candidates(node, doc):

            if (node.pos_ in ["PROPN", "NOUN", "X"]) and node.text not in self.punctuation: # and node.pos_ not in ["PRON"]:
                term_candidates = TermCandidate(doc[node.i:node.i + 1])

                for child in node.children:

                    start_index = min(node.i, child.i)
                    end_index = max(node.i, child.i)

                    if child.dep_ == "compound" or child.dep_.endswith('mod'): #child.dep_ == "amod":
                        subtree_tokens = list(child.subtree)
                        if all([c.dep_ == "compound" for c in subtree_tokens]):
                            start_index = min([c.i for c in subtree_tokens])
                        term_candidates.append(doc[start_index:end_index + 1])

                        if not child.dep_.endswith('mod'):
                            term_candidates.append(doc[start_index:start_index + 1])
                        excluded_children.append(child)

                    if (child.dep_ == "prep" and child.text == "of") or child.dep_ == "prt":
                        end_index = max([c.i for c in child.subtree])
                        term_candidates.append(doc[start_index:end_index + 1])

                    if child.text.lower() == "the": # include definite determiner in NP
                        start_index = min([c.i for c in child.subtree])
                        term_candidates.append(doc[start_index:end_index + 1])

                candidates.append(term_candidates)

            for child in node.children:
                if child in excluded_children:
                    continue
                get_candidates(child, doc)

        get_candidates(root, doc)

        if self.single_term and sent not in reduce(lambda a, candidate: a + candidate.variations, candidates, []):
            # For single-term extraction, add full term if not already added
            candidates.append(TermCandidate(sent))

        return candidates
