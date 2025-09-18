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

Citation
-----

If you use this software, please cite it as below and rference this implementation:
### GraMi

```
@article{elseidy2015grami,
  title={Grami: Frequent subgraph and pattern mining in a single large graph},
  author={Elseidy, Mohammed and Abdelhamid, Ehab and Skiadopoulos, Spiros and Kalnis, Panos},
  year={2015},
  publisher={Association for Computing Machinery (ACM)}
}
```

### SuGraMi, SoGraMi, PaGraMi (SoPaGraMi)
```
@article{nguyen2020fast,
  title={Fast and scalable algorithms for mining subgraphs in a single large graph},
  author={Nguyen, Lam BQ and Vo, Bay and Le, Ngoc-Thao and Snasel, Vaclav and Zelinka, Ivan},
  journal={Engineering Applications of Artificial Intelligence},
  volume={90},
  pages={103539},
  year={2020},
  publisher={Elsevier}
}
```