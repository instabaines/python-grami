"""Pattern representation used by the miners.

This module defines a lightweight `Pattern` class storing vertex labels,
edges and whether the pattern is directed. It also computes a canonical
key used for pattern equivalence (via a DFS-based canonicalization).
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Set
from canonical import _CanonDFSHelper
from graph import Edge


class Pattern:
    """Container for a graph pattern.

    Attributes:
    - directed: whether edges have direction.
    - vlabels: list of vertex labels (indexed by vertex id in the pattern).
    - edges: list of `Edge` objects describing the pattern topology.
    - key: canonical key (tuple) used to compare/uniquely identify patterns.
    """

    def __init__(self, vlabels: List[str], edges: List[Edge], directed: bool):
        self.directed = directed
        self.vlabels = list(vlabels)
        self.edges = list(edges)
        self.key = self._canonical_key()

    def _canonical_key(self) -> Tuple:
        """Return a canonical, comparable key for this pattern.

        The canonical key is produced by `_CanonDFSHelper` and consists of
        the directed flag plus the canonical DFS enumeration. This key is
        used throughout the codebase to deduplicate isomorphic patterns.
        """
        canon = _CanonDFSHelper(self.directed, self.vlabels, self.edges).canonical_code()
        return (self.directed, tuple(canon))

    def num_nodes(self) -> int:
        """Return number of vertices in the pattern."""
        return len(self.vlabels)

    def edge_set(self) -> Set[Tuple[int, int, Optional[str]]]:
        """Return a set of edges suitable for quick membership checks.

        For undirected patterns the edges are normalized so that (u,v)
        is always ordered (min, max).
        """
        if self.directed:
            return {(e.u, e.v, e.label) for e in self.edges}
        return {(min(e.u, e.v), max(e.u, e.v), e.label) for e in self.edges}
