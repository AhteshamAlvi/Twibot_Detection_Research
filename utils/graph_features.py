from pathlib import Path

import networkx as nx
import pandas as pd


GRAPH_FEATURE_COLS = [
    "graph_in_degree",
    "graph_out_degree",
    "graph_degree_total",
    "graph_ff_ratio",
    "graph_reciprocity_count",
    "graph_reciprocity_ratio",
    "graph_pagerank",
    "graph_component_size",
    "graph_neighbor_avg_in_degree",
    "graph_neighbor_avg_out_degree",
    "graph_neighbor_avg_degree_total",
    "graph_neighbor_avg_ff_ratio",
    "graph_neighbor_std_degree_total",
    "graph_ego_node_count",
    "graph_ego_edge_count",
    "graph_ego_density",
]


GRAPH_DIAGNOSTIC_COLS = [
    "neighbor_bot_ratio",
    "follower_bot_ratio",
    "following_bot_ratio",
    "ego_bot_ratio",
]

def inspect_csv_columns(path, nrows=5):
    preview = pd.read_csv(path, nrows=nrows, low_memory=False)
    print(path)
    print(preview.columns.tolist())
    display(preview)
    return preview


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

    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)

    edges = edges.dropna(subset=["source_id", "target_id"])
    edges = edges[edges["source_id"] != ""]
    edges = edges[edges["target_id"] != ""]
    edges = edges[edges["source_id"] != edges["target_id"]]

    edges = edges.drop_duplicates(subset=["source_id", "target_id"])

    if dataset_name is not None:
        edges["dataset"] = dataset_name

    if "relation" not in edges.columns:
        edges["relation"] = "follows"

    return edges[["dataset", "source_id", "target_id", "relation"]]


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
    edges = edges_df.copy()

    edge_pairs = set(zip(edges["source_id"], edges["target_id"]))

    reciprocal_sources = []

    for source, target in edge_pairs:
        if (target, source) in edge_pairs:
            reciprocal_sources.append(source)

    reciprocity = (
        pd.Series(reciprocal_sources, name="user_id")
        .value_counts()
        .rename_axis("user_id")
        .reset_index(name="graph_reciprocity_count")
    )

    out_degree = (
        edges.groupby("source_id")
        .size()
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


def build_networkx_graph(edges_df, directed=True):
    graph_type = nx.DiGraph() if directed else nx.Graph()

    graph = nx.from_pandas_edgelist(
        edges_df,
        source="source_id",
        target="target_id",
        create_using=graph_type,
    )

    return graph


def compute_pagerank_features(edges_df, max_edges=500_000):
    edges = edges_df.copy()

    if len(edges) > max_edges:
        edges = edges.sample(max_edges, random_state=42)

    graph = build_networkx_graph(edges, directed=True)

    pagerank = nx.pagerank(graph, max_iter=100)

    return (
        pd.DataFrame.from_dict(
            pagerank,
            orient="index",
            columns=["graph_pagerank"],
        )
        .reset_index()
        .rename(columns={"index": "user_id"})
    )


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

    neighbor_rows = edges.merge(
        degree_lookup,
        left_on="target_id",
        right_index=True,
        how="left",
    )

    neighbor_agg = (
        neighbor_rows
        .groupby("source_id")
        .agg(
            graph_neighbor_avg_in_degree=("graph_in_degree", "mean"),
            graph_neighbor_avg_out_degree=("graph_out_degree", "mean"),
            graph_neighbor_avg_degree_total=("graph_degree_total", "mean"),
            graph_neighbor_avg_ff_ratio=("graph_ff_ratio", "mean"),
            graph_neighbor_std_degree_total=("graph_degree_total", "std"),
        )
        .reset_index()
        .rename(columns={"source_id": "user_id"})
    )

    neighbor_agg["graph_neighbor_std_degree_total"] = (
        neighbor_agg["graph_neighbor_std_degree_total"].fillna(0)
    )

    return neighbor_agg


def compute_ego_density_features(edges_df, max_nodes_for_exact=200_000):
    """
    Computes 1-hop ego density per node.

    For very large graphs, this may be slow. Use on smaller datasets first,
    or run on sampled/filtered graphs.
    """

    graph = build_networkx_graph(edges_df, directed=False)

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


def compute_graph_features(edges_df, dataset_name=None, include_pagerank=True, include_ego=True):
    edges = clean_edges(edges_df, dataset_name=dataset_name)

    degree_features = compute_degree_features(edges)
    reciprocity_features = compute_reciprocity_features(edges)
    component_features = compute_component_features(edges)
    neighbor_features = compute_neighbor_aggregate_features(edges, degree_features)

    graph_features = degree_features.merge(
        reciprocity_features,
        on="user_id",
        how="left",
    )

    graph_features = graph_features.merge(
        component_features,
        on="user_id",
        how="left",
    )

    graph_features = graph_features.merge(
        neighbor_features,
        on="user_id",
        how="left",
    )

    if include_pagerank:
        pagerank_features = compute_pagerank_features(edges)
        graph_features = graph_features.merge(
            pagerank_features,
            on="user_id",
            how="left",
        )
    else:
        graph_features["graph_pagerank"] = 0

    if include_ego:
        ego_features = compute_ego_density_features(edges)
        graph_features = graph_features.merge(
            ego_features,
            on="user_id",
            how="left",
        )
    else:
        graph_features["graph_ego_node_count"] = 0
        graph_features["graph_ego_edge_count"] = 0
        graph_features["graph_ego_density"] = 0

    graph_features["dataset"] = dataset_name

    for col in GRAPH_FEATURE_COLS:
        if col in graph_features.columns:
            graph_features[col] = graph_features[col].fillna(0)

    return graph_features[["dataset", "user_id"] + [c for c in GRAPH_FEATURE_COLS if c in graph_features.columns]]


def merge_graph_features(base_features_df, graph_features_df):
    base = base_features_df.copy()
    graph = graph_features_df.copy()

    base["user_id"] = base["user_id"].astype(str)
    graph["user_id"] = graph["user_id"].astype(str)

    if "dataset" in base.columns and "dataset" in graph.columns:
        merged = base.merge(graph, on=["dataset", "user_id"], how="left")
    else:
        merged = base.merge(graph, on="user_id", how="left")

    for col in GRAPH_FEATURE_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged


def save_graph_frame(df, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def available_graph_feature_cols(df):
    return [c for c in GRAPH_FEATURE_COLS if c in df.columns]