python-grami
=============

A small Python implementation of simple graph mining utilities (MNI-based)
for educational and experimental use. The repository provides lightweight
representations for labeled graphs, pattern canonicalization, embedding
enumeration and simple miners.

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

Usage Notes
-----------

- The code is intentionally minimal and focuses on clarity over
  performance. It is suitable for teaching and small-scale experiments.
- `SoPaGraMiMiner` uses simple heuristics and edge-type pruning to reduce
  the search space; `SuGraMiMiner` is the simpler reference implementation.

Next Steps
----------

- Add tests under a `tests/` directory for regression and CI.
- Add packaging (pyproject.toml) if you want to install the library.
- Replace the brute-force embedding enumerator with a more optimized
  VF2-like matcher for larger graphs.

License
-------

MIT-style (not included). Use according to your needs.