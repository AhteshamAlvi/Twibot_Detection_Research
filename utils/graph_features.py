from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import duckdb
import json
from IPython.display import display

GRAPH_FEATURE_COLS = [
    "graph_in_degree",                 # number of incoming edges (followers)
    "graph_out_degree",                # number of outgoing edges (following)
    "graph_degree_total",              # in + out
    "graph_ff_ratio",                  # in_degree / (out_degree + 1)
    "graph_reciprocity_count",         # how many mutual follows this user has
    "graph_reciprocity_ratio",         # reciprocity_count / (out_degree + 1)
    "graph_pagerank",                  # PageRank score (NaN if user got sampled out)
    "graph_pagerank_sampled",          # 1 if pagerank was computed on a sampled subgraph
    "graph_component_size",            # size of connected component (undirected)
    "graph_neighbor_avg_in_degree",    # mean in_degree across both in- and out-neighbors
    "graph_neighbor_avg_out_degree",
    "graph_neighbor_avg_degree_total",
    "graph_neighbor_avg_ff_ratio",
    "graph_neighbor_std_degree_total",
    "graph_ego_node_count",            # NaN if graph too large to compute exactly
    "graph_ego_edge_count",            # NaN if graph too large
    "graph_ego_density",               # NaN if graph too large
]

# IDs are always strings — see utils.base_features for rationale.
_STRING_ID_COLS = ["user_id", "tweet_id", "in_reply_to_user_id", "source_id", "target_id"]

GRAPH_DIAGNOSTIC_COLS = [
    "neighbor_bot_ratio",
    "follower_bot_ratio",
    "following_bot_ratio",
    "ego_bot_ratio",
]

# ================================
# DEBUG / INSPECTION UTILITIES
# ================================
def inspect_csv_columns(path, nrows=5):
    preview = pd.read_csv(path, nrows=nrows, low_memory=False)
    print(path)
    print(preview.columns.tolist())
    display(preview)
    return preview


# ================================
# EDGE CONSTRUCTION (DATASET LOADERS)
# ================================
def load_cresci_edges(raw_dataset_dir, dataset_name):
    """
    Loads Cresci-style followers.csv and friends.csv files.

    Standard meaning:
        source_id -> target_id
        source_id follows target_id
    """

    raw_dataset_dir = Path(raw_dataset_dir)
    edge_frames = []

    for subset_dir in raw_dataset_dir.iterdir():
        if not subset_dir.is_dir():
            continue

        friends_path = subset_dir / "friends.csv"
        followers_path = subset_dir / "followers.csv"

        if friends_path.exists():
            friends = pd.read_csv(friends_path, low_memory=False)
            friends.columns = friends.columns.str.lower()

            print(f"{dataset_name} | {subset_dir.name} | friends columns:", friends.columns.tolist())

            if {"source_id", "target_id"}.issubset(friends.columns):
                temp = friends[["source_id", "target_id"]].copy()
            elif {"user_id", "friend_id"}.issubset(friends.columns):
                temp = friends[["user_id", "friend_id"]].copy()
                temp = temp.rename(columns={
                    "user_id": "source_id",
                    "friend_id": "target_id",
                })
            elif friends.shape[1] >= 2:
                temp = friends.iloc[:, :2].copy()
                temp.columns = ["source_id", "target_id"]
            else:
                continue

            temp["relation"] = "friend"
            temp["subset"] = subset_dir.name
            edge_frames.append(temp)

        if followers_path.exists():
            followers = pd.read_csv(followers_path, low_memory=False)
            followers.columns = followers.columns.str.lower()

            print(f"{dataset_name} | {subset_dir.name} | followers columns:", followers.columns.tolist())

            if {"source_id", "target_id"}.issubset(followers.columns):
                temp = followers[["source_id", "target_id"]].copy()
            elif {"follower_id", "user_id"}.issubset(followers.columns):
                temp = followers[["follower_id", "user_id"]].copy()
                temp = temp.rename(columns={
                    "follower_id": "source_id",
                    "user_id": "target_id",
                })
            elif followers.shape[1] >= 2:
                temp = followers.iloc[:, :2].copy()
                temp.columns = ["source_id", "target_id"]
            else:
                continue

            temp["relation"] = "follower"
            temp["subset"] = subset_dir.name
            edge_frames.append(temp)

    if not edge_frames:
        return pd.DataFrame(columns=["dataset", "source_id", "target_id", "relation"])

    edges = pd.concat(edge_frames, ignore_index=True)
    edges = clean_edges(edges, dataset_name=dataset_name)

    return edges

def load_twibot20_edges(raw_dataset_dir, dataset_name="twibot_2020"):
    raw_dataset_dir = Path(raw_dataset_dir)

    files = ["train.json", "test.json", "dev.json"]
    edge_rows = []

    for fname in files:
        path = raw_dataset_dir / fname

        if not path.exists():
            print(f"Missing {fname}, skipping")
            continue

        print(f"Loading {fname}...")

        with open(path, "r") as f:
            data = json.load(f)

        for record in data:
            user_id = str(record.get("ID") or record.get("id") or record.get("user_id"))
            neighbor = record.get("neighbor")

            if not neighbor:
                continue

            # followers: f -> user
            for f in neighbor.get("follower", []):
                edge_rows.append({
                    "source_id": str(f),
                    "target_id": user_id,
                    "relation": "follower"
                })

            # following: user -> g
            for g in neighbor.get("following", []):
                edge_rows.append({
                    "source_id": user_id,
                    "target_id": str(g),
                    "relation": "following"
                })

    if not edge_rows:
        return pd.DataFrame(columns=["dataset", "source_id", "target_id", "relation"])

    edges = pd.DataFrame(edge_rows)

    print("Twibot-20 raw edges:", len(edges))

    edges = clean_edges(edges, dataset_name=dataset_name)

    print("Twibot-20 cleaned edges:", len(edges))

    return edges

def load_twibot22_edges(edge_path, labeled_ids, dataset_name="twibot_2022"):
    edge_path = Path(edge_path)

    print("Loading Twibot-22 edges with DuckDB...")

    con = duckdb.connect()

    # Register labeled IDs as a table
    labeled_df = pd.DataFrame({"user_id": labeled_ids.astype(str)})
    con.register("labeled_users", labeled_df)

    query = f"""
        SELECT
            source_id,
            target_id,
            relation
        FROM read_csv_auto('{edge_path}')
        WHERE relation IN ('following', 'followers', 'followed')
          AND (
              source_id IN (SELECT user_id FROM labeled_users)
           OR target_id IN (SELECT user_id FROM labeled_users)
          )
    """

    edges = con.execute(query).df()

    print("Filtered Twibot-22 edges:", len(edges))

    edges = clean_edges(edges, dataset_name=dataset_name)

    return edges

# ================================
# EDGE CONSTRUCTION (INTERACTIONS + COMBINATION)
# ================================
def build_interaction_edges(tweets_path, dataset_name):
    # Use DuckDB for large files
    con = duckdb.connect()
    tweets_path = str(Path(tweets_path))
    query = f"""
        SELECT
            CAST(user_id AS VARCHAR) AS source_id,
            CAST(in_reply_to_user_id AS VARCHAR) AS target_id,
            'reply' AS relation
        FROM read_csv_auto('{tweets_path}')
        WHERE in_reply_to_user_id IS NOT NULL
          AND TRIM(CAST(in_reply_to_user_id AS VARCHAR)) != ''
          AND TRIM(CAST(in_reply_to_user_id AS VARCHAR)) != '0'
    """
    edges = con.execute(query).df()
    con.close()
    return clean_edges(edges, dataset_name=dataset_name)

def combine_edges(follow_df, interaction_df):
    frames = [df for df in [follow_df, interaction_df] if df is not None and len(df) > 0]
    if not frames:
        return pd.DataFrame(columns=["dataset", "source_id", "target_id", "relation"])
    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate on (source_id, target_id) — keep first occurrence
    combined = combined.drop_duplicates(subset=["source_id", "target_id"])
    return combined

def load_edges_for_dataset(dataset_name, raw_paths, preprocessed_paths=None):
    """
    raw_paths:          dict with key "raw_graph" pointing to the raw edge data dir/file
    preprocessed_paths: dict with keys "users" and "tweets" pointing to pre-processed CSVs
                        Required for cresci_2017; optional for others (adds interaction edges)
    """
    follow_df = pd.DataFrame(columns=["dataset", "source_id", "target_id", "relation"])

    if dataset_name == "cresci_2015":
        follow_df = load_cresci_edges(raw_paths["raw_graph"], dataset_name)

    elif dataset_name == "cresci_2017":
        pass  # No follow graph exists for cresci_2017

    elif dataset_name == "twibot_2020":
        follow_df = load_twibot20_edges(raw_paths["raw_graph"], dataset_name)

    elif dataset_name == "twibot_2022":
        users = pd.read_csv(preprocessed_paths["users"], usecols=["user_id"], low_memory=False)
        labeled_ids = users["user_id"].astype(str)
        follow_df = load_twibot22_edges(raw_paths["raw_graph"], labeled_ids, dataset_name)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Build interaction edges if tweets path provided
    interaction_df = pd.DataFrame(columns=["dataset", "source_id", "target_id", "relation"])
    if preprocessed_paths and preprocessed_paths.get("tweets"):
        interaction_df = build_interaction_edges(preprocessed_paths["tweets"], dataset_name)

    return combine_edges(follow_df, interaction_df)

# ================================
# EDGE CLEANING / STANDARDIZATION
# ================================
def clean_edges(edges_df, dataset_name=None):
    """
    Standardizes edge dataframe.

    Required meaning:
        source_id -> target_id
        source_id follows target_id
    """

    edges = edges_df.copy()

    required_cols = {"source_id", "target_id"}
    missing = required_cols - set(edges.columns)

    if missing:
        raise ValueError(f"Missing required edge columns: {missing}")

    for col in ("source_id", "target_id"):
        s = edges[col].astype("string")
        # Strip ".0" stringification artefacts from float-upcast int IDs
        mask = s.notna() & s.str.match(r"^-?\d+\.0+$")
        s = s.mask(mask, s.str.replace(r"\.0+$", "", regex=True))
        edges[col] = s

    edges = edges.dropna(subset=["source_id", "target_id"])
    edges = edges[~edges["source_id"].isin(["", "nan", "None"])]
    edges = edges[~edges["target_id"].isin(["", "nan", "None"])]
    edges = edges[edges["source_id"] != edges["target_id"]]

    edges = edges.drop_duplicates(subset=["source_id", "target_id"])

    if dataset_name is not None:
        edges["dataset"] = dataset_name

    if "relation" not in edges.columns:
        edges["relation"] = "follows"

    return edges[["dataset", "source_id", "target_id", "relation"]]

# ================================
# GRAPH CONSTRUCTION
# ================================
def build_networkx_graph(edges_df, directed=True):
    graph_type = nx.DiGraph() if directed else nx.Graph()

    graph = nx.from_pandas_edgelist(
        edges_df,
        source="source_id",
        target="target_id",
        create_using=graph_type,
    )

    return graph

# ================================
# CORE GRAPH FEATURE COMPUTATIONS
# ================================
def compute_degree_features(edges_df):
    edges = edges_df.copy()

    in_degree = (
        edges.groupby("target_id")
        .size()
        .rename("graph_in_degree")
        .reset_index()
        .rename(columns={"target_id": "user_id"})
    )

    out_degree = (
        edges.groupby("source_id")
        .size()
        .rename("graph_out_degree")
        .reset_index()
        .rename(columns={"source_id": "user_id"})
    )

    features = in_degree.merge(out_degree, on="user_id", how="outer").fillna(0)

    features["graph_in_degree"] = features["graph_in_degree"].astype(int)
    features["graph_out_degree"] = features["graph_out_degree"].astype(int)

    features["graph_degree_total"] = (
        features["graph_in_degree"] + features["graph_out_degree"]
    )

    features["graph_ff_ratio"] = (
        features["graph_in_degree"] / (features["graph_out_degree"] + 1)
    )

    return features


def compute_reciprocity_features(edges_df):
    """
    Per-source count of mutual edges, computed via a self-merge instead of
    a Python-level for-loop over every edge pair.
    """
    edges = edges_df[["source_id", "target_id"]].copy()
    edges = edges.drop_duplicates()

    swap = edges.rename(columns={"source_id": "_t", "target_id": "_s"})
    recip = edges.merge(
        swap,
        left_on=["source_id", "target_id"],
        right_on=["_s", "_t"],
        how="inner",
    )

    reciprocity = (
        recip.groupby("source_id").size()
        .rename("graph_reciprocity_count")
        .reset_index()
        .rename(columns={"source_id": "user_id"})
    )

    out_degree = (
        edges.groupby("source_id").size()
        .rename("graph_out_degree")
        .reset_index()
        .rename(columns={"source_id": "user_id"})
    )

    result = out_degree.merge(reciprocity, on="user_id", how="left")
    result["graph_reciprocity_count"] = result["graph_reciprocity_count"].fillna(0)

    result["graph_reciprocity_ratio"] = (
        result["graph_reciprocity_count"] / (result["graph_out_degree"] + 1)
    )

    return result[
        [
            "user_id",
            "graph_reciprocity_count",
            "graph_reciprocity_ratio",
        ]
    ]


def compute_pagerank_features(edges_df, max_edges=500_000):
    """
    Compute PageRank.

    If we exceed `max_edges`, we sample down to fit memory. Users who get
    sampled OUT will have NaN pagerank (not 0) so downstream code can
    distinguish "low pagerank" from "we don't know."

    The companion `graph_pagerank_sampled` flag records whether sampling
    happened at all (it's a per-graph fact, not per-user, but we attach it
    per-user for ease of merging downstream).
    """
    edges = edges_df.copy()

    sampled = len(edges) > max_edges
    if sampled:
        edges = edges.sample(max_edges, random_state=42)

    graph = build_networkx_graph(edges, directed=True)
    pagerank = nx.pagerank(graph, max_iter=100)

    out = (
        pd.DataFrame.from_dict(
            pagerank,
            orient="index",
            columns=["graph_pagerank"],
        )
        .reset_index()
        .rename(columns={"index": "user_id"})
    )
    out["graph_pagerank_sampled"] = int(sampled)
    return out


def compute_component_features(edges_df):
    graph = build_networkx_graph(edges_df, directed=False)

    rows = []

    for component in nx.connected_components(graph):
        size = len(component)

        for node in component:
            rows.append({
                "user_id": str(node),
                "graph_component_size": size,
            })

    return pd.DataFrame(rows)


def compute_neighbor_aggregate_features(edges_df, degree_features):
    """
    Aggregate degree statistics across a user's neighborhood, where
    "neighborhood" includes both inbound (followers) and outbound (followees).

    The previous implementation only walked outbound edges, so a user with
    only inbound edges (popular but inactive accounts) got NaN aggregates
    that were then filled with 0 — indistinguishable from "neighbors with
    degree 0."
    """
    edges = edges_df.copy()
    degrees = degree_features.copy()

    degree_lookup = degrees.set_index("user_id")[
        [
            "graph_in_degree",
            "graph_out_degree",
            "graph_degree_total",
            "graph_ff_ratio",
        ]
    ]

    # User → outbound neighbor (target_id is the neighbor)
    out_rows = edges.merge(
        degree_lookup,
        left_on="target_id", right_index=True, how="left",
    )[["source_id"] + list(degree_lookup.columns)].rename(
        columns={"source_id": "user_id"}
    )

    # User → inbound neighbor (source_id is the neighbor)
    in_rows = edges.merge(
        degree_lookup,
        left_on="source_id", right_index=True, how="left",
    )[["target_id"] + list(degree_lookup.columns)].rename(
        columns={"target_id": "user_id"}
    )

    neighbor_rows = pd.concat([out_rows, in_rows], ignore_index=True)

    neighbor_agg = (
        neighbor_rows
        .groupby("user_id")
        .agg(
            graph_neighbor_avg_in_degree=("graph_in_degree", "mean"),
            graph_neighbor_avg_out_degree=("graph_out_degree", "mean"),
            graph_neighbor_avg_degree_total=("graph_degree_total", "mean"),
            graph_neighbor_avg_ff_ratio=("graph_ff_ratio", "mean"),
            graph_neighbor_std_degree_total=("graph_degree_total", "std"),
        )
        .reset_index()
    )

    # std() is NaN when there's only one neighbor — that's "no spread", so 0
    # is a reasonable fill. Everything else stays NaN if it's missing.
    neighbor_agg["graph_neighbor_std_degree_total"] = (
        neighbor_agg["graph_neighbor_std_degree_total"].fillna(0)
    )

    return neighbor_agg


def compute_ego_density_features(edges_df, max_nodes_for_exact=200_000):
    """
    Computes 1-hop ego density per node.

    For very large graphs, this may be slow. Use on smaller datasets first,
    or run on sampled/filtered graphs.

    If the graph exceeds max_nodes_for_exact nodes, returns zero-filled ego
    features for all nodes rather than hanging indefinitely.
    """

    graph = build_networkx_graph(edges_df, directed=False)

    if graph.number_of_nodes() > max_nodes_for_exact:
        print(
            f"  Graph too large ({graph.number_of_nodes():,} nodes > "
            f"{max_nodes_for_exact:,}) — returning NaN ego features"
        )
        return pd.DataFrame({
            "user_id": pd.Series(list(graph.nodes()), dtype=str),
            "graph_ego_node_count": np.nan,
            "graph_ego_edge_count": np.nan,
            "graph_ego_density": np.nan,
        })

    rows = []

    for node in graph.nodes:
        neighbors = set(graph.neighbors(node))
        ego_nodes = neighbors | {node}

        ego_node_count = len(ego_nodes)

        if ego_node_count <= 1:
            ego_edge_count = 0
            ego_density = 0
        else:
            subgraph = graph.subgraph(ego_nodes)
            ego_edge_count = subgraph.number_of_edges()
            possible_edges = ego_node_count * (ego_node_count - 1) / 2
            ego_density = ego_edge_count / possible_edges if possible_edges > 0 else 0

        rows.append({
            "user_id": str(node),
            "graph_ego_node_count": ego_node_count,
            "graph_ego_edge_count": ego_edge_count,
            "graph_ego_density": ego_density,
        })

    return pd.DataFrame(rows)

def compute_node2vec_embeddings(
    edges_df,
    dataset_name,
    dimensions=64,
    walk_length=10,
    num_walks=10,        # 🔥 reduced default (50 is too large)
    window=5,
    max_edges=500_000,
):
    """
    Computes Node2Vec embeddings for a graph.

    Returns:
        DataFrame with columns:
        [dataset, user_id, emb_0 ... emb_{dimensions-1}]
    """

    from node2vec import Node2Vec  # lazy import — heavy dependency

    if len(edges_df) == 0:
        return pd.DataFrame(
            columns=["dataset", "user_id"] + [f"emb_{i}" for i in range(dimensions)]
        )

    edges = edges_df.copy()

    # 🔥 FIX: sample BEFORE graph construction
    if len(edges) > max_edges:
        print(f"[{dataset_name}] Sampling edges for Node2Vec: {len(edges):,} → {max_edges:,}")
        edges = edges.sample(max_edges, random_state=42)

    # Build graph
    G = nx.from_pandas_edgelist(
        edges,
        source="source_id",
        target="target_id",
        create_using=nx.DiGraph(),
    )

    print(f"[{dataset_name}] Graph for embeddings: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # 🔥 safer defaults (avoid explosion)
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=2,
    )

    model = node2vec.fit(window=window, min_count=1)

    # Build dataframe
    rows = []
    for node in model.wv.index_to_key:
        rows.append({
            "dataset": dataset_name,
            "user_id": str(node),
            **{f"emb_{i}": val for i, val in enumerate(model.wv[node])}
        })

    emb_df = pd.DataFrame(rows)

    return emb_df

# ================================
# FEATURE ORCHESTRATION (MAIN PIPELINE)
# ================================
def compute_graph_features(edges_df, dataset_name=None, include_pagerank=True, include_ego=True):
    """
    Compute the canonical graph feature set for a single dataset.

    NaN semantics:
      - Degree-based features (in/out/total/ff_ratio, reciprocity, component
        size) are zero-filled because zero is meaningful (you have no edges
        of that direction).
      - PageRank, neighbor aggregates, and ego features stay NaN where they
        cannot be computed, so the model can distinguish "low value" from
        "we couldn't measure it." This matters across datasets because the
        ego/pagerank pipelines have different feasibility profiles.
    """
    edges = clean_edges(edges_df, dataset_name=dataset_name)

    degree_features = compute_degree_features(edges)
    reciprocity_features = compute_reciprocity_features(edges)
    component_features = compute_component_features(edges)
    neighbor_features = compute_neighbor_aggregate_features(edges, degree_features)

    graph_features = degree_features.merge(reciprocity_features, on="user_id", how="left")
    graph_features = graph_features.merge(component_features, on="user_id", how="left")
    graph_features = graph_features.merge(neighbor_features, on="user_id", how="left")

    if include_pagerank:
        pagerank_features = compute_pagerank_features(edges)
        graph_features = graph_features.merge(pagerank_features, on="user_id", how="left")
        graph_features["graph_pagerank_sampled"] = (
            graph_features["graph_pagerank_sampled"].fillna(0).astype("Int8")
        )
    else:
        graph_features["graph_pagerank"] = np.nan
        graph_features["graph_pagerank_sampled"] = pd.NA

    if include_ego:
        ego_features = compute_ego_density_features(edges)
        graph_features = graph_features.merge(ego_features, on="user_id", how="left")
    else:
        graph_features["graph_ego_node_count"] = np.nan
        graph_features["graph_ego_edge_count"] = np.nan
        graph_features["graph_ego_density"] = np.nan

    graph_features["dataset"] = dataset_name

    # Fill ZERO-meaningful features with 0; leave NaN-meaningful as NaN.
    zero_fill_cols = [
        "graph_in_degree",
        "graph_out_degree",
        "graph_degree_total",
        "graph_ff_ratio",
        "graph_reciprocity_count",
        "graph_reciprocity_ratio",
        "graph_component_size",
    ]
    for col in zero_fill_cols:
        if col in graph_features.columns:
            graph_features[col] = graph_features[col].fillna(0)

    return graph_features[
        ["dataset", "user_id"] + [c for c in GRAPH_FEATURE_COLS if c in graph_features.columns]
    ]

# ================================
# MERGE WITH BASE FEATURES
# ================================
def merge_graph_features(base_features_df, graph_features_df):
    """
    Left-join graph features onto base features. Both sides MUST carry a
    `dataset` column — without it, joining on user_id alone across the
    four-dataset frame would create a cross-product. We assert rather than
    silently fall back.

    Zero-meaningful columns (degrees, reciprocity, component size) are
    zero-filled where missing. The remaining graph columns stay NaN where
    missing, preserving the "we don't know" signal.
    """
    base = base_features_df.copy()
    graph = graph_features_df.copy()

    assert "dataset" in base.columns, "base features missing `dataset` column"
    assert "dataset" in graph.columns, "graph features missing `dataset` column"

    base["user_id"] = base["user_id"].astype(str)
    graph["user_id"] = graph["user_id"].astype(str)

    merged = base.merge(graph, on=["dataset", "user_id"], how="left")

    zero_fill_cols = [
        "graph_in_degree",
        "graph_out_degree",
        "graph_degree_total",
        "graph_ff_ratio",
        "graph_reciprocity_count",
        "graph_reciprocity_ratio",
        "graph_component_size",
    ]
    for col in zero_fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged

def merge_embeddings(base_df, embeddings_df):
    """
    Left-join node2vec embeddings. As with `merge_graph_features`, both sides
    must carry `dataset` to avoid a cross-product across the multi-dataset
    frame.

    NOTE on cross-dataset use: node2vec embeddings are computed independently
    per dataset (and on a sampled subgraph for large graphs), so embedding
    spaces are NOT aligned across datasets. Treat `emb_*` as auxiliary
    features for within-dataset analysis only — exclude them from the
    cross-era model unless you have an alignment step.
    """
    base = base_df.copy()
    emb = embeddings_df.copy()

    assert "dataset" in base.columns, "base frame missing `dataset` column"
    assert "dataset" in emb.columns, "embeddings frame missing `dataset` column"

    base["user_id"] = base["user_id"].astype(str)
    emb["user_id"] = emb["user_id"].astype(str)

    merged = base.merge(emb, on=["dataset", "user_id"], how="left")

    emb_cols = [c for c in merged.columns if c.startswith("emb_")]

    if emb_cols:
        merged[emb_cols] = merged[emb_cols].fillna(0)

    return merged

# ================================
# OUTPUT / PERSISTENCE
# ================================
def save_graph_frame(df, out_path):
    """
    Save a graph-feature frame. Parquet is preferred (preserves dtypes and
    is much faster to load); falls back to CSV by suffix.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


def available_graph_feature_cols(df):
    return [c for c in GRAPH_FEATURE_COLS if c in df.columns]

# ================================
# DIAGNOSTICS / EDA (LABEL-LEAKING)
# ================================
def compute_label_diagnostics(edges_df, labels_df):
    """
    EDA-only diagnostics.

    Do NOT use these as model-training features because they use true labels.
    """

    edges = edges_df.copy()
    labels = labels_df[["user_id", "label"]].copy()

    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)
    labels["user_id"] = labels["user_id"].astype(str)

    labels["is_bot"] = labels["label"].astype(str).str.lower().eq("bot").astype(int)

    source_neighbors = edges.merge(
        labels[["user_id", "is_bot"]],
        left_on="target_id",
        right_on="user_id",
        how="left",
    )

    following_bot_ratio = (
        source_neighbors
        .groupby("source_id")["is_bot"]
        .mean()
        .reset_index()
        .rename(columns={
            "source_id": "user_id",
            "is_bot": "following_bot_ratio",
        })
    )

    target_neighbors = edges.merge(
        labels[["user_id", "is_bot"]],
        left_on="source_id",
        right_on="user_id",
        how="left",
    )

    follower_bot_ratio = (
        target_neighbors
        .groupby("target_id")["is_bot"]
        .mean()
        .reset_index()
        .rename(columns={
            "target_id": "user_id",
            "is_bot": "follower_bot_ratio",
        })
    )

    diagnostics = following_bot_ratio.merge(
        follower_bot_ratio,
        on="user_id",
        how="outer",
    )

    diagnostics["neighbor_bot_ratio"] = diagnostics[
        ["following_bot_ratio", "follower_bot_ratio"]
    ].mean(axis=1)

    return diagnostics

