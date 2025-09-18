python-grami
=============

A python implementation of Grami Based algorithm for mining subgraphs from single large graphs

Structure
---------

- `src/graph.py` - `Edge` and `DataGraph` simple graph container and loaders.
- `src/pattern.py` - `Pattern` class and canonical key generation wrapper.
- `src/canonical.py` - Internal DFS-based canonicalization helper.
- `src/embed.py` - Embedding enumeration and MNI support computation.
- `src/candidate.py` - Candidate generation (seed patterns and extensions).
- `src/miner.py` - Two miner classes: `SuGraMiMiner` and `SoPaGraMiMiner`.
- `src/test.py` - Small runnable example building a tiny graph and mining it.

Quickstart
----------

Requirements: Python 3.8+

To run the tiny example:

```powershell
python src\test.py
```

This will build a small undirected triangle graph and run the default
`SuGraMiMiner`, printing discovered frequent patterns and simple stats.



License
-------

This project is licensed under the Apache License, Version 2.0. See
the `LICENSE` file for the full license text.
