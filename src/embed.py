"""Embedding enumeration and support computation.

This module implements backtracking-based exact embedding enumeration
and support computations used by the miners. It provides:
- `full_support_count`: count (optionally capped) of full embeddings
- `full_mni_embeddings`: enumerate all MNI embeddings (used as base)
- `mni_support`: compute Minimum Image-based support from embeddings
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from pattern import Pattern
from graph import DataGraph


class EmbeddingEnumerator:
    """Provides methods to enumerate embeddings of a pattern in a graph.

    The implementation uses simple backtracking with a domain per pattern
    vertex (nodes with matching label). The assignment order prefers
    small domains and high-degree pattern nodes to improve pruning.
    """

    def __init__(self, G: DataGraph):
        self.G = G

    def full_support_count(self, p: Pattern, cap: Optional[int] = None) -> int:
        """Return number of full (injective) embeddings of `p` in `G`.

        If `cap` is provided the search stops early once the count reaches
        the cap value (used for early-pruning scenarios).
        """
        domains = self._initial_domains(p)
        order = self._assignment_order(p, domains)
        assignment: Dict[int, int] = {}
        count = 0
        nbrs: Dict[int, List[Tuple[int, Optional[str], int]]] = defaultdict(list)
        for e in p.edges:
            nbrs[e.u].append((e.v, e.label, 1))
            if not p.directed:
                nbrs[e.v].append((e.u, e.label, 1))
            else:
                nbrs[e.v].append((e.u, e.label, 2))

        def consistent(u_p: int, u_g: int) -> bool:
            if p.vlabels[u_p] != self.G.vlabels[u_g]:
                return False
            for v_p, elab, d in nbrs[u_p]:
                if v_p in assignment:
                    v_g = assignment[v_p]
                    if d == 1:
                        if not self.G.has_edge(u_g, v_g, elab):
                            return False
                    else:
                        if not self.G.has_edge(v_g, u_g, elab):
                            return False
            return True

        def backtrack(i: int):
            nonlocal count
            if cap is not None and count >= cap:
                return
            if i == len(order):
                count += 1
                return
            u_p = order[i]
            for u_g in domains[u_p]:
                if u_g in assignment.values():
                    continue
                if consistent(u_p, u_g):
                    assignment[u_p] = u_g
                    backtrack(i + 1)
                    del assignment[u_p]

        backtrack(0)
        return count

    def full_mni_embeddings(self, p: Pattern) -> List[Dict[int, int]]:
        """Enumerate all injective embeddings of pattern `p` in `G`.

        Returns a list of dictionaries mapping pattern node -> graph node.
        """
        domains = self._initial_domains(p)
        order = self._assignment_order(p, domains)
        assignment: Dict[int, int] = {}
        solutions: List[Dict[int, int]] = []
        nbrs: Dict[int, List[Tuple[int, Optional[str], int]]] = defaultdict(list)
        for e in p.edges:
            nbrs[e.u].append((e.v, e.label, 1))
            if not p.directed:
                nbrs[e.v].append((e.u, e.label, 1))
            else:
                nbrs[e.v].append((e.u, e.label, 2))

        def consistent(u_p: int, u_g: int) -> bool:
            if p.vlabels[u_p] != self.G.vlabels[u_g]:
                return False
            for v_p, elab, d in nbrs[u_p]:
                if v_p in assignment:
                    v_g = assignment[v_p]
                    if d == 1:
                        if not self.G.has_edge(u_g, v_g, elab):
                            return False
                    elif d == 2:
                        if not self.G.has_edge(v_g, u_g, elab):
                            return False
            return True

        def backtrack(i: int):
            if i == len(order):
                solutions.append(dict(assignment))
                return
            u_p = order[i]
            for u_g in sorted(domains[u_p]):
                if u_g in assignment.values():
                    continue
                if consistent(u_p, u_g):
                    assignment[u_p] = u_g
                    backtrack(i + 1)
                    del assignment[u_p]

        backtrack(0)
        return solutions

    def mni_support(self, embeddings: List[Dict[int,int]], k: int) -> int:
        """Compute MNI (minimum image-based) support from embeddings.

        For each pattern node i we collect the distinct graph nodes it maps
        to across all embeddings and return the minimum cardinality among
        those sets.
        """
        if k == 0:
            return 0
        imgs: List[Set[int]] = [set() for _ in range(k)]
        for emb in embeddings:
            for pnode, gnode in emb.items():
                imgs[pnode].add(gnode)
        return min(len(s) for s in imgs)

    def _initial_domains(self, p: Pattern) -> Dict[int, Set[int]]:
        return {i: set(self.G.lab2nodes.get(lbl, set())) for i, lbl in enumerate(p.vlabels)}

    def _assignment_order(self, p: Pattern, domains: Dict[int, Set[int]]) -> List[int]:
        from collections import defaultdict
        deg = defaultdict(int)
        for e in p.edges:
            deg[e.u] += 1; deg[e.v] += 1
        # order by (smallest domain, highest pattern degree, stable id)
        return sorted(range(p.num_nodes()), key=lambda u: (len(domains[u]), -deg[u], u))
