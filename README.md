# Clustering on Actor-Movie Graph Dataset

Graph-based unsupervised learning project that groups actors and movies into clusters using heterogeneous random walk embeddings and K-Means clustering on a bipartite actor-movie network.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Custom Evaluation Metrics](#custom-evaluation-metrics)
- [Visualizations](#visualizations)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

---

## Overview

The goal is to discover natural groupings among actors and movies by leveraging the structure of their co-appearance network. Rather than relying on explicit content features (genre, ratings, etc.), the project treats the actor-movie relationship as a **bipartite graph** and learns node embeddings from the graph topology alone.

**Two clustering tasks are performed:**
1. **Actor Clustering** — group actors who tend to appear in similar movies
2. **Movie Clustering** — group movies that share similar casts

---

## Dataset

| Property | Value |
|----------|-------|
| Source file | `movie_actor_network.csv` |
| Total edges (actor-movie pairs) | 9,650 |
| Total nodes | 4,703 |
| — Actors | 3,411 |
| — Movies | 1,292 |
| Graph type | Bipartite (undirected) |
| Connected components | 1 (fully connected giant component) |

Actors are prefixed with `a` and movies with `m` to distinguish node types within the graph.

---

## Methodology

### 1. Graph Construction

The CSV is loaded into a **NetworkX** bipartite graph where each edge connects an actor node to a movie node. The resulting graph is fully connected (single giant component), ensuring all nodes are reachable.

### 2. Node Embedding via Heterogeneous Random Walks

**StellarGraph** is used to generate heterogeneous random walks that respect the bipartite node types:

| Meta-path | Description |
|-----------|-------------|
| `["movie", "actor", "movie"]` | movie-centric context |
| `["actor", "movie", "actor"]` | actor-centric context |

**Walk parameters:**
- Max walk length: 100
- Walks per node: 1
- Total walks generated: 4,703

### 3. Word2Vec Embeddings

The random walk sequences are treated as "sentences" and fed into **Gensim's Word2Vec** model to learn dense vector representations:

- Embedding dimension: **128**
- Window size: 5
- Output: one 128-dimensional vector per node (4,703 total)

### 4. Clustering

Node embeddings are split by type and clustered independently:

- **Actor embeddings** (3,411 × 128) → K-Means clustering
- **Movie embeddings** (1,292 × 128) → K-Means clustering

Candidate values for `k`: 3, 5, 10, 30, 50, 100, 200, 500

The optimal `k` is selected by maximizing a custom graph-aware cost metric (see below).

### 5. Dimensionality Reduction for Visualization

**t-SNE** is applied to project the high-dimensional embeddings to 2D for scatter plot visualization.

| Task | Perplexity | Iterations |
|------|-----------|------------|
| Actor clustering | 70 | 1,500 |
| Movie clustering | 150 | 5,000 |

---

## Custom Evaluation Metrics

Standard clustering metrics (silhouette score, Davies-Bouldin index) do not account for graph structure. Two graph-aware cost functions are defined:

### Cost1 — Intra-cluster Connectivity

```
Cost1 = (1/N) * sum( largest_connected_component_size / cluster_size )
```

Measures how well nodes in each cluster are connected to each other within the original graph.

### Cost2 — Intra-cluster Density

```
Cost2 = (1/N) * sum( sum_of_degrees / unique_neighbors_in_cluster )
```

Measures the density of relationships between nodes within each cluster.

### Final Score

```
Final Score = Cost1 × Cost2
```

The number of clusters `k` that maximizes this product is selected.

---

## Results

### Actor Clustering

| k | Cost Score (C1 × C2) |
|---|----------------------|
| **3** | **4.165** (selected) |
| 5 | 3.298 |
| 10 | 2.059 |
| 30 | 1.846 |
| 50 | 1.583 |
| 100 | 1.953 |
| 200 | 1.766 |
| 500 | 1.919 |

**Optimal k = 3** — Actor collaborations naturally segment into 3 distinct communities.

---

### Movie Clustering

| k | Cost Score (C1 × C2) |
|---|----------------------|
| 3 | 8.461 |
| 5 | 10.320 |
| 10 | 8.811 |
| 30 | 12.550 |
| **50** | **14.474** (selected) |
| 100 | 14.254 |
| 200 | 12.722 |
| 500 | 10.306 |

**Optimal k = 50** — Movies form 50 distinct clusters, reflecting more diverse grouping patterns compared to actors.

---

## Visualizations

| Visualization | Description |
|---------------|-------------|
| Bipartite graph plot | Initial network structure showing actor-movie connections |
| Actor cluster scatter plot | t-SNE 2D projection of 3,411 actors colored by cluster (k=3) |
| Movie cluster scatter plot | t-SNE 2D projection of 1,292 movies colored by cluster (k=50) |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `stellargraph` | Heterogeneous random walk generation on graphs |
| `gensim` | Word2Vec model for node embedding |
| `networkx==2.3` | Graph construction and analysis |
| `scikit-learn` | K-Means clustering and t-SNE |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualization |
| `tensorflow` | Backend dependency for StellarGraph |

Install dependencies:

```bash
pip install stellargraph gensim "networkx==2.3" scikit-learn pandas numpy matplotlib tensorflow
```

---

## How to Run

This project is designed to run in **Google Colab**.

1. Open `K_means_Clustering_on_actor_movie_graph_dataset_.ipynb` in Google Colab
2. Run the dependency installation cells at the top
3. When prompted, upload `movie_actor_network.csv` using the file upload widget
4. Run all remaining cells sequentially

The notebook will:
- Build the actor-movie bipartite graph
- Generate node embeddings via random walks + Word2Vec
- Run K-Means for multiple values of `k`
- Evaluate each configuration using the custom cost metric
- Produce t-SNE visualizations of the final clusters

---

## Key Takeaways

- **Graph structure alone** is sufficient to discover meaningful groupings — no content features needed
- Actors form **3 tight communities** likely reflecting different eras, genres, or production networks
- Movies form **50 diverse clusters**, suggesting a richer variety of collaborative patterns across the film industry
- The custom graph-aware metric (Cost1 × Cost2) outperforms generic clustering metrics for this network-based task
