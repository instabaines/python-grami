"""Simple runnable example for the package.

This small script builds a tiny undirected triangle graph and runs
the SuGraMi miner to demonstrate basic usage. It is not a full test
suite; it exists as a convenience entrypoint for quick local checks.
"""

from graph import DataGraph, Edge
from pattern import Pattern

from miner import SuGraMiMiner, SoPaGraMiMiner, SoGraMiHeuristics


if __name__ == "__main__":
    # Construct a tiny undirected triangle graph with vertex labels
    # labelled 'X','Y','Z'. The graph has three edges (0-1, 1-2, 2-0).
    vlabels = ["X", "Y", "Z"]
    edges = [Edge(0, 1), Edge(1, 2), Edge(2, 0)]
    G = DataGraph(directed=False, vlabels=vlabels, edges=edges)

    # Create a miner with minimum support 1 and run it in parallel
    miner = SuGraMiMiner(G, min_support=1, max_workers=2)
    results = miner.mine(parallel=True, max_size=3)

    # Pretty-print discovered frequent patterns and some statistics
    print(f"Found {len(results)} frequent patterns:")
    for key, rec in results.items():
        p: Pattern = rec['pattern']
        print("Pattern key:", key)
        print("  |V|=", p.num_nodes(), "|E|=", len(p.edges))
        print("  support:", rec['support'])
        print("  embeddings count:", len(rec['embeddings']))
        print()
