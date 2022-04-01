import json
import psycopg2
import os

MAX_DEPTH_CHAIN = 10
P_INSTANCE_OF = 31
P_SUBCLASS = 279

MAX_ITEMS_CACHE = 100000

conn = None
entity_cache = {}
chain_cache = {}

wikidata_instance = None


def get_wikidata_instance():
    global wikidata_instance

    if wikidata_instance is None:
        wikidata_instance = WikidataQueryController()

    return wikidata_instance


class WikidataQueryController:

    def __init__(self):
        self.conn = None

        self.cache = {
            "entity": {},
            "chain": {},
            "name": {}
        }

        self.init_database_connection(self._load_creds())

    def _get_cached_value(self, cache_type, key):
        return self.cache[cache_type][key]

    def _is_cached(self, cache_type, key):
        return key in self.cache[cache_type]

    def _add_to_cache(self, cache_type, key, value):
        if len(self.cache[cache_type]) < MAX_ITEMS_CACHE:
            self.cache[cache_type][key] = value

    def _load_creds(self):
        # get the credentials path from an env variable
        credsPath = os.environ.get("c3creds")
        if not credsPath or not os.path.isfile(credsPath):
            raise CredentialsNotFound()
        # load up the creds
        with open(credsPath) as credsFile:
            try:
                return json.load(credsFile)["data_sources"]["wikidata"]["location"]
            except:
                raise InvalidCredentials("...everything", "Your c3creds file doesn't seem to be formatted correctly.")

    def init_database_connection(self, path):
        self.conn = psycopg2.connect(path)

    def clear_cache(self):
        self.cache["entity"].clear()
        self.cache["chain"].clear()
        self.cache["name"].clear()

    def get_entities_from_alias(self, alias):
        c = self.conn.cursor()
        if self._is_cached("entity", alias):
            return self._get_cached_value("entity", alias).copy()

        query_alias = """select i.id, i.name, i.description, a.alias
            from item_alias_view a
            left join item i on a.item_id = i.id
            where a.lowercase_alias = %s"""

        c.execute(query_alias, [alias.lower()])
        fetched_rows = c.fetchall()

        self._add_to_cache("entity", alias, fetched_rows)
        return fetched_rows

    def get_instances_of(self, item_id, properties=[P_INSTANCE_OF, P_SUBCLASS], count=1000):
        query = f"select subject_id from item_relationship where object_id = %s and property_id in ({','.join([str(prop) for prop in properties])}) LIMIT %s"

        c = self.conn.cursor()
        c.execute(query, [item_id, count])

        res = c.fetchall()

        return [e[0] for e in res]

    def get_entity_name(self, item_id):
        if self._is_cached("name", item_id):
            return self._get_cached_value("name", item_id)

        c = self.conn.cursor()
        query = "SELECT name from item WHERE id=%s"
        c.execute(query, [item_id])
        res = c.fetchone()

        if res and len(res):
            if res[0] == None:
                self._add_to_cache("name", item_id, 'no label')
            else:
                self._add_to_cache("name", item_id, res[0])
        else:
            self._add_to_cache("name", item_id, '<none>')

        return self._get_cached_value("name", item_id)

    def get_entity(self, item_id):
        c = self.conn.cursor()
        query = """SELECT id, name, description from item
                   WHERE id=%s"""

        res = c.execute(query, [item_id])

        return res.fetchone()

    def get_children(self, item_id, limit=100):
        c = self.conn.cursor()
        query = """SELECT i.id, i.name, i.description from item as i
                   JOIN item_relationship as r on i.id=r.subject_id
                   WHERE r.object_id=%s and r.property_id IN (279,31) LIMIT %s"""

        res = c.execute(query, [item_id, limit])

        return res.fetchall()

    def get_parents(self, item_id, limit=100):
        c = self.conn.cursor()
        query = """SELECT i.id, i.name, i.description from item as i
                   JOIN item_relationship as r on i.id=r.object_id
                   WHERE r.subject_id=%s and r.property_id IN (279,31) LIMIT %s"""

        res = c.execute(query, [item_id, limit])

        return res.fetchall()

    def get_subclasses(self, item_id, max_depth=10):
        chain = []
        edges = []
        self._append_chain_elements(item_id, 0, chain, edges, max_depth, P_SUBCLASS, traverse_up=False)
        return [el[0] for el in chain]

    def get_categories(self, item_id, max_depth=10):
        chain = []
        edges = []
        self._append_chain_elements(item_id, 0, chain, edges, max_depth, [P_INSTANCE_OF, P_SUBCLASS])
        return [el[0] for el in chain]

    def get_chain(self, item_id, max_depth=10, property=P_INSTANCE_OF):
        chain = []
        edges = []
        self._append_chain_elements(item_id, 0, chain, edges, max_depth, property)
        return chain

    def get_recursive_edges(self, item_id):
        chain = []
        edges = []
        self._append_chain_elements(item_id, 0, chain, edges)
        return edges

    def _append_chain_elements(self, item_id, level=0, chain=[], edges=[], max_depth=10, property=P_INSTANCE_OF, traverse_up=True):
        properties = property
        if type(property) != list:
            properties = [property]

        if self._is_cached("chain", (item_id, max_depth)):
            chain += self._get_cached_value("chain", (item_id, max_depth)).copy()
            return

        # prevent infinite recursion
        if level >= max_depth:
            return

        c = self.conn.cursor()

        query = f"SELECT object_id, property_id from item_relationship where subject_id=%s and property_id IN ({','.join([str(prop) for prop in properties])})" if traverse_up \
                else f"SELECT subject_id, property_id from item_relationship where object_id=%s and property_id IN ({','.join([str(prop) for prop in properties])})"

        # set value for current item in order to prevent infinite recursion
        self._add_to_cache("chain", (item_id, max_depth), [])
        c.execute(query, [item_id])

        for target_item in c.fetchall():

            chain_ids = [el[0] for el in chain]

            if not (target_item[0] in chain_ids):
                chain += [(target_item[0], level + 1)]
                edges.append((item_id, target_item[0], target_item[1]))
                self._append_chain_elements(target_item[0], level=level + 1, chain=chain, edges=edges,
                                            max_depth=max_depth,
                                            property=property,
                                            traverse_up=traverse_up)

        self._add_to_cache("chain", (item_id, max_depth), chain)


if __name__ == '__main__':
    queryInstance = WikidataQueryController()

    queryInstance.init_database_connection()
    print(queryInstance.get_categories(13191, max_depth=1))
    print(queryInstance.get_categories(13191, max_depth=1))
