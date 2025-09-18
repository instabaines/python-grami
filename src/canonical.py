"""Canonicalization helper used to compute a canonical DFS code for
patterns. The canonical code is used to compare patterns for isomorphism
and to produce a stable key for deduplication.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Set, Dict
from graph import Edge


class _CanonDFSHelper:
    """Internal helper that enumerates DFS-based canonical codes.

    The algorithm starts from every possible seed edge and performs a
    deterministic DFS enumeration that produces a sequence; the lexicographically
    smallest sequence across seeds is returned as the canonical code.
    """

    def __init__(self, directed: bool, vlabels: List[str], edges: List[Edge]):
        self.directed = directed
        self.vlabels = vlabels
        self.n = len(vlabels)
        self.p_adj: List[List[Tuple[int, Optional[str], int]]] = [[] for _ in range(self.n)]
        self.edge_keys: Set[Tuple[int, int, Optional[str], int]] = set()
        for e in edges:
            if directed:
                self.p_adj[e.u].append((e.v, e.label, 1))
                self.p_adj[e.v].append((e.u, e.label, 2))
                self.edge_keys.add((e.u, e.v, e.label, 1))
            else:
                self.p_adj[e.u].append((e.v, e.label, 0))
                self.p_adj[e.v].append((e.u, e.label, 0))
                a, b = (e.u, e.v) if e.u <= e.v else (e.v, e.u)
                self.edge_keys.add((a, b, e.label, 0))

    def canonical_code(self) -> Tuple[Tuple, ...]:
        """Compute and return the canonical DFS code as a tuple of tuples.

        The method iterates seed edges and selects the lexicographically
        smallest DFS enumeration produced by `_dfs_enumerate`.
        """
        seeds: List[Tuple[int, int, Optional[str], int]] = []
        if self.directed:
            for (u, v, el, d) in sorted(self.edge_keys):
                if d == 1:
                    seeds.append((u, v, el, 1))
        else:
            for (u, v, el, d) in sorted(self.edge_keys):
                seeds.append((u, v, el, 0))

        best: Optional[Tuple[Tuple, ...]] = None
        for su, sv, el, df in seeds:
            seq = self._dfs_enumerate(su, sv, el, df)
            t = tuple(seq)
            if best is None or t < best:
                best = t
        return best or tuple()

    def _dfs_enumerate(self, su: int, sv: int, elab: Optional[str], dflag: int) -> List[Tuple]:
        visited_idx: Dict[int, int] = {}

        def assign(u: int) -> int:
            if u in visited_idx:
                return visited_idx[u]
            visited_idx[u] = len(visited_idx)
            return visited_idx[u]

        used_edges: Set[Tuple[int, int, Optional[str], int]] = set()
        code: List[Tuple] = []

        def norm_key(u: int, v: int, el: Optional[str], df: int) -> Tuple[int, int, Optional[str], int]:
            if self.directed:
                return (u, v, el, 1)
            a, b = (u, v) if u <= v else (v, u)
            return (a, b, el, 0)

        def push_edge(u: int, v: int, el: Optional[str], df: int):
            uidx = assign(u); vidx = assign(v)
            code.append((uidx, vidx, self.vlabels[u], el or "", self.vlabels[v], df))
            used_edges.add(norm_key(u, v, el, df))

        push_edge(su, sv, elab, dflag)

        while True:
            if used_edges >= self.edge_keys:
                break
            frontier: List[Tuple[int, int, Optional[str], int, int, int]] = []
            for u, uidx in list(visited_idx.items()):
                for v, el, df in self.p_adj[u]:
                    key = norm_key(u, v, el, df)
                    if key in used_edges:
                        continue
                    to_seen = (v in visited_idx)
                    frontier.append((u, v, el, df, uidx, visited_idx.get(v, 10**9)))
            if not frontier:
                break
            frontier.sort(key=lambda t: (
                t[4],
                self.vlabels[t[0]],
                t[2] or "",
                0 if t[5] != 10**9 else 1,
                self.vlabels[t[1]] if t[5] != 10**9 else "~",
                t[3],
                t[5],
            ))
            u, v, el, df, _, _ = frontier[0]
            push_edge(u, v, el, df)
        return code
