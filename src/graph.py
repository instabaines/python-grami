"""Lightweight graph model used by the mining algorithms.

This module defines two simple primitives:
- `Edge`: immutable dataclass representing a labeled (u,v) edge.
- `DataGraph`: adjacency-based graph container with helpers used by
  the pattern miners and embedding enumerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


@dataclass(frozen=True)
class Edge:
    """Immutable edge record.

    Attributes:
    - u, v: integer node ids
    - label: optional edge label (can be None)
    """
    u: int
    v: int
    label: Optional[str] = None


class DataGraph:
    """Simple adjacency-list graph used as input to the miners.

    The class stores both forward adjacency (`adj`) and reverse adjacency
    (`rev`) lists. For undirected graphs both structures contain the same
    neighbor entries (i.e., edges are stored in both directions).
    `lab2nodes` maps vertex label -> set(node ids) for quick domain
    computation during embedding enumeration.
    """

    @classmethod
    def from_lg(cls, path: str, directed: bool = False) -> "DataGraph":
        """Load a graph from a simple .lg-like format.

        Expected file lines:
        - `v <id> [label]` defines a vertex (label optional)
        - `e <u> <v> [elabel]` defines an edge (label optional)
        """
        vlabels: Dict[int, str] = {}
        edges: List[Edge] = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                tag = parts[0].lower()
                if tag == 'v':
                    idx = int(parts[1])
                    lab = parts[2] if len(parts) > 2 else ""
                    vlabels[idx] = lab
                elif tag == 'e':
                    u = int(parts[1])
                    v = int(parts[2])
                    elab = parts[3] if len(parts) > 3 else None
                    edges.append(Edge(u, v, elab))
        max_idx = max(vlabels) if vlabels else -1
        labels_list = [vlabels.get(i, "") for i in range(max_idx + 1)]
        return cls(directed, labels_list, edges)

    def __init__(self, directed: bool, vlabels: List[str], edges: List[Edge]):
        self.directed = directed
        self.vlabels = vlabels
        n = len(vlabels)
        # forward adjacency: list of (neighbor, label)
        self.adj: List[List[Tuple[int, Optional[str]]]] = [[] for _ in range(n)]
        # reverse adjacency: useful for directed graphs
        self.rev: List[List[Tuple[int, Optional[str]]]] = [[] for _ in range(n)]
        # adjacency sets indexed by neighbor -> set(labels) for quick checks
        self.adj_set: List[Dict[int, Set[Optional[str]]]] = [defaultdict(set) for _ in range(n)]
        self.rev_set: List[Dict[int, Set[Optional[str]]]] = [defaultdict(set) for _ in range(n)]
        for e in edges:
            self.adj[e.u].append((e.v, e.label))
            self.adj_set[e.u][e.v].add(e.label)
            self.rev[e.v].append((e.u, e.label))
            self.rev_set[e.v][e.u].add(e.label)
            if not directed:
                # for undirected graphs add symmetric entries
                self.adj[e.v].append((e.u, e.label))
                self.adj_set[e.v][e.u].add(e.label)
                self.rev[e.u].append((e.v, e.label))
                self.rev_set[e.u][e.v].add(e.label)
        # map label -> set of nodes with that label
        self.lab2nodes: Dict[str, Set[int]] = defaultdict(set)
        for i, lab in enumerate(vlabels):
            self.lab2nodes[lab].add(i)

    def edge_type_counts(self) -> Dict[Tuple[str, str, Optional[str], int], int]:
        """Count occurrences of each (label,label,edge_label,directed_flag) type.

        Returns a mapping useful for seed selection and frequency pruning.
        The last element of the tuple is 1 for directed edges and 0 for
        undirected (normalized) edges.
        """
        counts: Dict[Tuple[str, str, Optional[str], int], int] = defaultdict(int)
        if self.directed:
            for u in range(len(self.vlabels)):
                lu = self.vlabels[u]
                for v, el in self.adj[u]:
                    lv = self.vlabels[v]
                    counts[(lu, lv, el, 1)] += 1
        else:
            for u in range(len(self.vlabels)):
                lu = self.vlabels[u]
                for v, el in self.adj[u]:
                    if u > v:
                        continue
                    lv = self.vlabels[v]
                    a, b = (lu, lv) if lu <= lv else (lv, lu)
                    counts[(a, b, el, 0)] += 1
        return counts

    def has_edge(self, u: int, v: int, label: Optional[str]) -> bool:
        """Return True if an edge (u->v) exists; if `label` is provided it
        must match the edge label.
        """
        for w, lab in self.adj[u]:
            if w == v and (label is None or lab == label):
                return True
        return False
