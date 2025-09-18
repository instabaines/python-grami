"""Graph mining entry points and helpers.

This module implements two miners used in experiments:
- `SuGraMiMiner`: straightforward subgraph miner using MNI support.
- `SoPaGraMiMiner`: an improved variant that prunes by frequent edge
  types and (optionally) uses simple heuristics for neighbor ordering.

The file also contains helpers to materialize subgraphs from embeddings
and a small worker function used when evaluating embeddings in parallel
via `ProcessPoolExecutor`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from collections import defaultdict

from graph import DataGraph, Edge
from pattern import Pattern
from embed import EmbeddingEnumerator
from candidate import CandidateGenerator


class SuGraMiMiner:
    """Basic subgraph miner using MNI support.

    The miner performs a BFS-like expansion of candidate patterns where
    each frontier is evaluated for support. Patterns meeting `min_support`
    are kept and extended until `max_size` is reached (if provided).
    """

    def __init__(self, G: DataGraph, min_support: int, max_workers: int = os.cpu_count() or 2):
        self.G = G
        self.min_support = min_support
        self.embedder = EmbeddingEnumerator(G)
        self.candgen = CandidateGenerator(G)
        self.max_workers = max_workers

    def mine(self, parallel: bool = True, max_size: Optional[int] = None) -> Dict[Tuple, Dict]:
        """Run the mining process and return discovered frequent patterns.

        Returned value is a mapping from pattern `key` -> dict with keys:
        - `pattern`: the Pattern object
        - `support`: MNI support
        - `full_support`: raw number of injective embeddings
        - `embeddings`: list of embeddings used to compute support
        """
        results: Dict[Tuple, Dict] = {}
        frontier: List[Pattern] = list(self.candgen.seed_patterns())

        while frontier:
            eval_results = self._evaluate_patterns(frontier, parallel)
            next_frontier: List[Pattern] = []
            for p, embeddings in eval_results:
                supp = self.embedder.mni_support(embeddings, p.num_nodes())
                if supp >= self.min_support:
                    full_supp = self.embedder.full_support_count(p)
                    results[p.key] = {'pattern': p, 'support': supp, 'full_support': full_supp, 'embeddings': embeddings}
                    if max_size is None or p.num_nodes() < max_size:
                        for q in self.candgen.extensions(p, embeddings):
                            if q.key not in results:
                                next_frontier.append(q)
            seen: Set[Tuple] = set()
            dedup: List[Pattern] = []
            for q in next_frontier:
                if q.key not in seen:
                    seen.add(q.key)
                    dedup.append(q)
            frontier = dedup
        return results

    def _evaluate_patterns(self, patterns: List[Pattern], parallel: bool) -> List[Tuple[Pattern, List[Dict[int, int]]]]:
        """Evaluate a batch of patterns and return their embeddings.

        If `parallel` is True a process pool is used to evaluate each pattern
        in parallel (useful for expensive embedding enumeration on large
        graphs).
        """
        if not parallel or self.max_workers <= 1 or len(patterns) == 1:
            return [(p, self.embedder.full_mni_embeddings(p)) for p in patterns]
        out: List[Tuple[Pattern, List[Dict[int, int]]]] = []
        payload = (self.G.directed, self.G.vlabels, self.G.adj, self.G.rev)
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            fut2p = {ex.submit(_worker_embeddings, payload, p.vlabels, p.edges): p for p in patterns}
            for fut in as_completed(fut2p):
                p = fut2p[fut]
                embeddings = fut.result()
                out.append((p, embeddings))
        return out


def _worker_embeddings(payload, p_vlabels: List[str], p_edges: List[Edge]) -> List[Dict[int, int]]:
    """Worker function used by process pool to compute embeddings.

    The process reconstructs a minimal `DataGraph` object from the
    serialized payload and runs `EmbeddingEnumerator.full_mni_embeddings`.
    """
    directed, vlabels, adj, rev = payload
    G = DataGraph(directed, vlabels, [])
    G.adj = adj
    G.rev = rev
    G.lab2nodes = defaultdict(set)
    for i, lab in enumerate(vlabels):
        G.lab2nodes[lab].add(i)
    p = Pattern(p_vlabels, p_edges, directed)
    return EmbeddingEnumerator(G).full_mni_embeddings(p)


def subgraph_from_embedding(G: DataGraph, p: Pattern, emb: Dict[int, int], induced: bool = False) -> Tuple[List[int], List[Edge]]:
    """Materialize a concrete subgraph from `emb` mapping pattern->graph.

    If `induced` is True the returned edges are all edges among the
    selected graph nodes; otherwise they follow the pattern's edges.
    """
    nodes = sorted(set(emb.values()))
    node_set = set(nodes)
    edges: List[Edge] = []
    if induced:
        seen = set()
        for u in nodes:
            for v, lab in G.adj[u]:
                if v in node_set:
                    if G.directed:
                        edges.append(Edge(u, v, lab))
                    else:
                        a, b = (u, v) if u < v else (v, u)
                        key = (a, b, lab)
                        if key not in seen:
                            seen.add(key)
                            edges.append(Edge(a, b, lab))
    else:
        for e in p.edges:
            u = emb[e.u]; v = emb[e.v]
            edges.append(Edge(u, v, e.label))
    return nodes, edges


def materialize_all_embeddings(G: DataGraph, p: Pattern, embeddings: List[Dict[int, int]], induced: bool = False) -> List[Tuple[List[int], List[Edge]]]:
    return [subgraph_from_embedding(G, p, emb, induced=induced) for emb in embeddings]


class SoGraMiHeuristics:
    """Small helper providing ordering/pruning heuristics used by SoPaGraMi.

    The heuristics are intentionally simple: label rarity and degree-based
    ordering/pruning to prioritize rare labels and higher-degree nodes.
    """

    def __init__(self, G: DataGraph):
        self.G = G
        self.lab_freq: Dict[str, int] = {lab: len(nodes) for lab, nodes in G.lab2nodes.items()}
        self.deg = [len(G.adj[i]) for i in range(len(G.vlabels))]

    def label_rarity(self, lab: str) -> int:
        return self.lab_freq.get(lab, 0)

    def neighbor_order(self, u_g: int) -> List[Tuple[int, Optional[str]]]:
        neigh = list(self.G.adj[u_g])
        neigh.sort(key=lambda t: (self.label_rarity(self.G.vlabels[t[0]]), -self.deg[t[0]]))
        return neigh

    def degree_prune(self, need_deg: int, cand_deg: int) -> bool:
        return cand_deg >= need_deg


class SoPaGraMiMiner(SuGraMiMiner):
    """Variant of SuGraMi that pre-filters edge types and uses heuristics.

    This miner first computes frequent edge types and only starts from
    those seed edge types; it can also pass neighbor-ordering heuristics
    to the candidate generator to reduce expansion.
    """

    def __init__(self, G: DataGraph, min_support: int, max_workers: int = os.cpu_count() or 2,
                 use_sorting: bool = True):
        super().__init__(G, min_support, max_workers)
        self.heur = SoGraMiHeuristics(G) if use_sorting else None

    def mine(self, parallel: bool = True, max_size: Optional[int] = None) -> Dict[Tuple, Dict]:
        results: Dict[Tuple, Dict] = {}
        counts = self.G.edge_type_counts()
        allowed_edge_types: Set[Tuple[str, str, Optional[str], int]] = {
            k for k, c in counts.items() if c >= self.min_support
        }
        seeds: List[Tuple[Pattern, int]] = []
        for (lu, lv, el, dflag), c in counts.items():
            if (lu, lv, el, dflag) not in allowed_edge_types:
                continue
            if dflag == 1:
                p = Pattern([lu, lv], [Edge(0, 1, el)], True)
            else:
                p = Pattern([lu, lv], [Edge(0, 1, el)], False)
            seeds.append((p, c))
        seeds.sort(key=lambda pc: pc[1], reverse=True)
        frontier: List[Pattern] = [p for p, _ in seeds]

        while frontier:
            eval_results = self._evaluate_patterns(frontier, parallel)
            next_frontier: List[Pattern] = []
            for p, embeddings in eval_results:
                supp = self.embedder.mni_support(embeddings, p.num_nodes())
                if supp >= self.min_support:
                    full_supp = self.embedder.full_support_count(p)
                    results[p.key] = {'pattern': p, 'support': supp, 'full_support': full_supp, 'embeddings': embeddings}
                    if max_size is None or p.num_nodes() < max_size:
                        for q in self.candgen.extensions(p, embeddings, heur=self.heur if hasattr(self.candgen, 'extensions') else None, allowed_edge_types=allowed_edge_types):
                            if q.key not in results:
                                next_frontier.append(q)
            seen: Set[Tuple] = set()
            dedup: List[Pattern] = []
            for q in next_frontier:
                if q.key not in seen:
                    seen.add(q.key)
                    dedup.append(q)
            frontier = dedup
        return results
