"""Add USE for entire raw text to vmware corpus."""
from vmware.vector.use import encode


_use_mapping = {
  "properties": {
    "first_line_use": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_0": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_1": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_2": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_3": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_4": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_5": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_6": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_7": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_8": {
        "type": "dense_vector",
        "dims": 512
    },
    "long_remaining_lines_use_9": {
        "type": "dense_vector",
        "dims": 512
    }
  }
}


def _process_use_remaining_lines(doc_source):
    """Process USE data on long passages and the first line."""
    doc_source["first_line_use"] = encode(doc_source["first_line"]).tolist()
    long_remaining_lines = [line for line in doc_source['remaining_lines'] if len(line) > 20]
    for idx, line in enumerate(long_remaining_lines):
        if idx < 10:
            doc_source[f"long_remaining_lines_use_{idx}"] = encode(line).tolist()
    return doc_source


mapping = _use_mapping
enrichment = _process_use_remaining_lines
