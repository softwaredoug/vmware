import elasticsearch.helpers
import json
import os
from time import perf_counter
from elasticsearch.exceptions import NotFoundError


def resp_msg(msg, resp, throw=True):
    print('{} [Status: {}]'.format(msg, resp.status_code))
    if resp.status_code >= 400:
        print(resp.text)
        if throw:
            raise RuntimeError(resp.text)


version_field = "enrich_version"


def enrich(es, index, enrich_fn, mapping, version, workers=2):
    """Incrementally enrich documents not yet at the specified version."""
    search_scroll_body = {
        "query": {
            "match": {
                version_field: version - 1
            }
        }
    }
    count = es.count(index=index, body=search_scroll_body)
    print(f"Enriching {count['count']} documents in {index}")

    if mapping is not None:
        es.indices.put_mapping(index=index, body=mapping)

    def scanner():
        start = perf_counter()
        for idx, doc in enumerate(elasticsearch.helpers.scan(es, index=index, scroll='5m',
                                                             size=1000,
                                                             query=search_scroll_body)):

            curr_version = int(doc['_source'][version_field])

            assert version == curr_version + 1

            doc['_source'] = enrich_fn(doc['_source'])
            doc["_source"][version_field] = version
            try:
                yield {
                    "_op_type": "update",
                    "_index": index,
                    "_id": doc['_id'],
                    "doc": doc['_source']
                }

                # return self.es.update(index=index, id=doc['_id'], body={"doc": doc['_source']})
            except NotFoundError:
                print(f"Document {doc['_id']} not found, skipping")

            if idx % 100 == 0:
                print(f"Enriched {idx} documents -- {perf_counter() - start}")
                print("--------")

    resps = elasticsearch.helpers.parallel_bulk(es, scanner(), thread_count=workers,
                                                chunk_size=100, request_timeout=120)
    for success, resp in resps:
        if not success:
            print("Failure -- ")
            print(resp)
    es.indices.refresh(index=index)


class ElasticResp():
    def __init__(self, resp):
        self.status_code = 400
        if 'acknowledged' in resp and resp['acknowledged']:
            self.status_code = 200
        else:
            self.status_code = resp['status']
            self.text = json.dumps(resp, indent=2)


def index_documents(es, index, doc_src):

    def bulkDocs(doc_src):
        for doc in doc_src:
            if 'id' not in doc:
                raise ValueError("Expecting docs to have field 'id' that uniquely identifies document")
            doc[version_field] = 0
            addCmd = {"_index": index,
                      "_id": doc['id'],
                      "_source": doc}
            yield addCmd

    elasticsearch.helpers.bulk(es, bulkDocs(doc_src), chunk_size=100, request_timeout=120)
    es.indices.refresh(index=index)


def rebuild(es, index, doc_src, configs_dir='.'):
    """ Reload a configuration on disk for each search engine
        (Solr a configset, Elasticsearch a json file)
        and reindex
    """
    es.indices.delete(index=index, ignore=[400, 404])

    cfg_json_path = os.path.join(configs_dir, "%s_settings.json" % index)
    with open(cfg_json_path) as src:
        settings = json.load(src)
        es.indices.create(index, body=settings)

    index_documents(es, index, doc_src=doc_src)
