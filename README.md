# 3D Objects Retrieval using LSH

- `src` directory contain the main LSH class implementation which has member functions for hashing objects and retrieving objects from the tables. `src` also contains different feature classes, and code for extracting object features from a mesh.
- `compute_hash_tables.py` contain wrapper functions around LSH, that also compute object features from mesh, and then store to / retrieve from the LSH tables. Example for using wrapper functions is given in `compute_hash_tables.py` at the end.
- Note: An initial framework was taken from `SparseLSH`, but currently only the storage class (wrapper around Python dictionary) is being used from the library.

