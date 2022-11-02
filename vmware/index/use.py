"""Add USE for entire raw text to vmware corpus."""
import vmware.vector.use

_use_mapping = {
  "properties": {
    "raw_text_use": {
        "type": "dense_vector",
        "dims": 512
    }
  }
}


def _process_use(doc_source):
    """Process USE data"""
    doc_source['raw_text_use'] = vmware.vector.use.encode(doc_source['raw_text']).tolist()
    return doc_source


mapping = _use_mapping
enrichment = _process_use
