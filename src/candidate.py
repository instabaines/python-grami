"""Generate candidate pattern extensions used during mining.

This module implements a small candidate generator that produces seed
patterns (single-edge patterns) and extends patterns by adding edges or
vertices based on embeddings and an rmpath heuristic derived from the
pattern's canonical code.
"""

from __future__ import annotations

from typing import Iterator, List, Set, Tuple, Optional, Dict, TYPE_CHECKING
from graph import Edge, DataGraph
from pattern import Pattern
from canonical import _CanonDFSHelper

if TYPE_CHECKING:
    from miner import SoGraMiHeuristics


class CandidateGenerator:
    """Produce initial seeds and one-edge extensions for a pattern.

    The `seed_patterns` method yields unique one-edge patterns present in
    the data graph (labels + edge label). The `extensions` method takes a
    pattern and a list of embeddings and yields possible single-edge
    extensions. Extensions are deduplicated using the pattern canonical
    key.
    """

    def __init__(self, G: DataGraph):
        self.G = G

    def seed_patterns(self) -> Iterator[Pattern]:
        seen: Set[Tuple] = set()
        if self.G.directed:
            for u in range(len(self.G.vlabels)):
                lu = self.G.vlabels[u]
                for v, elab in self.G.adj[u]:
                    lv = self.G.vlabels[v]
                    tpl = (lu, lv, elab, True)
                    if tpl in seen:
                        continue
                    seen.add(tpl)
                    yield Pattern([lu, lv], [Edge(0, 1, elab)], True)
        else:
            for u in range(len(self.G.vlabels)):
                lu = self.G.vlabels[u]
                for v, elab in self.G.adj[u]:
                    if u >= v:
                        continue
                    lv = self.G.vlabels[v]
                    tpl = (min(lu, lv), max(lu, lv), elab, False)
                    if tpl in seen:
                        continue
                    seen.add(tpl)
                    yield Pattern([lu, lv], [Edge(0, 1, elab)], False)

    def _rmpath(self, p: Pattern) -> List[int]:
        """Compute the right-most path (rmpath) from the canonical code.

        The rmpath is used to determine where to attach new edges when
        generating extensions (standard trick from graph mining literature).
        """
        code = _CanonDFSHelper(p.directed, p.vlabels, p.edges).canonical_code()
        if not code:
            return list(range(p.num_nodes()))
        seen: Set[int] = set()
        parent: Dict[int, int] = {}
        for frm_idx, to_idx, *_ in code:
            if frm_idx not in seen:
                seen.add(frm_idx)
            if to_idx not in seen:
                parent[to_idx] = frm_idx
                seen.add(to_idx)
        r = max({frm for frm, *_ in code} | {to for _, to, *_ in code})
        path = [r]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def extensions(
        self,
        p: Pattern,
        embeddings: List[Dict[int, int]],
        heur: Optional["SoGraMiHeuristics"] = None,
        allowed_edge_types: Optional[Set[Tuple[str, str, Optional[str], int]]] = None,
    ) -> Iterator[Pattern]:
        """Yield candidate one-edge extensions for pattern `p`.

        - `embeddings` are MNI embeddings of `p` in the graph and are used
          to guide where new edges can be attached.
        - `heur` (optional) can provide vertex ordering and degree pruning
          heuristics to reduce search.
        - `allowed_edge_types` is an optional pre-filter of edge types
          (label-label-edgeLabel-directedFlag) used by SoPaGraMi.
        """
        produced: Set[Tuple] = set()
        existing = p.edge_set()
        rmpath = self._rmpath(p) or list(range(p.num_nodes()))
        rm = rmpath[-1]
        ancestors = rmpath[:-1]

        def pat_deg(u_p: int) -> int:
            return sum(1 for e in p.edges if e.u == u_p or e.v == u_p)

        for emb in embeddings:
            u_p = rm
            u_g = emb.get(u_p)
            if u_g is not None:
                for w_p in reversed(ancestors):
                    w_g = emb.get(w_p)
                    if w_g is None:
                        continue
                    if not p.directed:
                        if self.G.has_edge(u_g, w_g, None) or self.G.has_edge(w_g, u_g, None):
                            labs = [lab for (v2, lab) in self.G.adj[u_g] if v2 == w_g]
                            if not labs:
                                labs = [lab for (v2, lab) in self.G.adj[w_g] if v2 == u_g]
                            for elab in labs:
                                a, b = (min(u_p, w_p), max(u_p, w_p))
                                if (a, b, elab) in existing:
                                    continue
                                if allowed_edge_types is not None:
                                    lu, lv = p.vlabels[u_p], p.vlabels[w_p]
                                    a, b = (lu, lv) if lu <= lv else (lv, lu)
                                    if (a, b, elab, 0) not in allowed_edge_types:
                                        continue
                                q = Pattern(list(p.vlabels), list(p.edges) + [Edge(u_p, w_p, elab)], False)
                                if q.key not in produced:
                                    produced.add(q.key)
                                    yield q
                    else:
                        if self.G.has_edge(u_g, w_g, None):
                            labs = [lab for (v2, lab) in self.G.adj[u_g] if v2 == w_g]
                            for elab in labs:
                                if (u_p, w_p, elab) in existing:
                                    continue
                                if allowed_edge_types is not None:
                                    lu, lv = p.vlabels[u_p], p.vlabels[w_p]
                                    if (lu, lv, elab, 1) not in allowed_edge_types:
                                        continue
                                q = Pattern(list(p.vlabels), list(p.edges) + [Edge(u_p, w_p, elab)], True)
                                if q.key not in produced:
                                    produced.add(q.key)
                                    yield q
                        if self.G.has_edge(w_g, u_g, None):
                            labs = [lab for (v2, lab) in self.G.adj[w_g] if v2 == u_g]
                            for elab in labs:
                                if (w_p, u_p, elab) in existing:
                                    continue
                                if allowed_edge_types is not None:
                                    lu, lv = p.vlabels[w_p], p.vlabels[u_p]
                                    if (lu, lv, elab, 1) not in allowed_edge_types:
                                        continue
                                q = Pattern(list(p.vlabels), list(p.edges) + [Edge(w_p, u_p, elab)], True)
                                if q.key not in produced:
                                    produced.add(q.key)
                                    yield q

            grow_verts = ([rm] + ancestors) if heur else rmpath
            for u_p in grow_verts:
                u_g = emb.get(u_p)
                if u_g is None:
                    continue
                out_neigh = heur.neighbor_order(u_g) if heur else self.G.adj[u_g]
                for v_g, elab in out_neigh:
                    if v_g in emb.values():
                        continue
                    if heur and not heur.degree_prune(pat_deg(u_p) + 1, len(self.G.adj[v_g])):
                        continue
                    lv = self.G.vlabels[v_g]
                    new_vid = p.num_nodes()
                    if allowed_edge_types is not None:
                        lu = p.vlabels[u_p]; lv2 = lv
                        if p.directed:
                            if (lu, lv2, elab, 1) not in allowed_edge_types:
                                continue
                        else:
                            a, b = (lu, lv2) if lu <= lv2 else (lv2, lu)
                            if (a, b, elab, 0) not in allowed_edge_types:
                                continue
                    q = Pattern(list(p.vlabels) + [lv], list(p.edges) + [Edge(u_p, new_vid, elab)], p.directed)
                    if q.key not in produced:
                        produced.add(q.key)
                        yield q

                if self.G.directed:
                    in_neigh = self.G.rev[u_g]
                    if heur:
                        in_neigh = sorted(in_neigh, key=lambda t: (
                            heur.label_rarity(self.G.vlabels[t[0]]), -len(self.G.adj[t[0]])
                        ))
                    for v_g, elab in in_neigh:
                        if v_g in emb.values():
                            continue
                        if heur and not heur.degree_prune(pat_deg(u_p) + 1, len(self.G.adj[v_g])):
                            continue
                        lv = self.G.vlabels[v_g]
                        new_vid = p.num_nodes()
                        if allowed_edge_types is not None:
                            lu2 = lv; lv_p = p.vlabels[u_p]
                            if (lu2, lv_p, elab, 1) not in allowed_edge_types:
                                continue
                        q = Pattern(list(p.vlabels) + [lv], list(p.edges) + [Edge(new_vid, u_p, elab)], True)
                        if q.key not in produced:
                            produced.add(q.key)
                            yield q
