import path  # noqa: F401

from vmware.index.colocations import scan_queries, scan_index

if __name__ == "__main__":
    scan_queries()
    scan_index(index='vmware')
