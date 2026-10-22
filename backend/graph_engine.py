"""
Graph Engine for AML Network Topology Visualization using NetworkX and Plotly.
"""

import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from backend.dataset_manager import dataset_manager


def generate_topology_graph(df: "pd.DataFrame | None" = None, top_n_nodes: int = 15) -> dict:
    """
    Generates an interactive Plotly graph showing customer nodes and transaction edges.

    `df`, if provided, is used directly (this is how main.py calls it, passing
    the currently-active dataset). If omitted, falls back to
    dataset_manager.get_active_data(). Previously this function only accepted
    `top_n_nodes` as its first positional parameter, so main.py's
    `generate_transaction_graph(df)` call silently bound the whole DataFrame
    to `top_n_nodes` instead of ignoring it -- corrupting every downstream
    slice (`df.head(top_n_nodes * 3)`, `cust_list[:top_n_nodes]`) and crashing
    on any non-empty dataset.
    """
    if df is None:
        df = dataset_manager.get_active_data()

    if df.empty:
        return {"error": "No dataset loaded"}

    # Identify sender and receiver columns
    sender_col = next((c for c in ['customer_id', 'sender_id', 'source_customer'] if c in df.columns), None)
    receiver_col = next((c for c in ['beneficiary_id', 'receiver_id', 'target_customer'] if c in df.columns), None)
    amount_col = next((c for c in ['amount', 'transaction_amount'] if c in df.columns), 'amount')

    G = nx.DiGraph()

    if sender_col and receiver_col and sender_col in df.columns and receiver_col in df.columns:
        # Build network from real transaction pairs
        sample_df = df.head(top_n_nodes * 3)
        for _, row in sample_df.iterrows():
            u = str(row[sender_col])
            v = str(row[receiver_col])
            amt = float(row.get(amount_col, 1000.0))
            G.add_edge(u, v, weight=amt)
    else:
        # Fallback: Build customer network based on top transactional entities
        cust_list = df[sender_col].dropna().unique()[:top_n_nodes] if sender_col else [f"CUST_{i:05d}" for i in range(1, top_n_nodes + 1)]
        for i in range(len(cust_list) - 1):
            G.add_edge(str(cust_list[i]), str(cust_list[i+1]), weight=5000.0)
            if i % 2 == 0 and i + 2 < len(cust_list):
                G.add_edge(str(cust_list[i]), str(cust_list[i+2]), weight=12000.0)

    # Layout calculation
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # Extract edge coordinates
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#64748B'),
        hoverinfo='none',
        mode='lines',
        showlegend=False,
    )

    # Extract node coordinates and properties
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)
        node_text.append(f"Customer: {node}<br>Connections: {degree}")
        
        # Color coding: High degree/connectivity gets highlighted.
        # Palette chosen for higher contrast between tiers (the previous
        # red/orange pair was hard to tell apart at marker size) and to
        # avoid a red/green pairing that's ambiguous for colorblind viewers.
        if degree >= 3:
            node_colors.append('#DC2626')  # High connectivity (red)
        elif degree == 2:
            node_colors.append('#D97706')  # Medium connectivity (amber)
        else:
            node_colors.append('#0F766E')  # Low connectivity (teal)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        textfont=dict(color='#0F172A', size=11, family='Arial, sans-serif'),
        hovertext=node_text,
        showlegend=False,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=22,
            line=dict(width=2, color='#FFFFFF')
        )
    )

    # Explicit legend: three invisible marker traces, one per risk tier, so
    # the meaning of node color isn't left for the viewer to guess. The
    # actual node_trace above has showlegend=False to avoid duplicating a
    # single "trace 0" legend entry.
    legend_traces = [
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=color),
            name=label, showlegend=True,
        )
        for label, color in [
            ("High connectivity (3+ links)", '#DC2626'),
            ("Medium connectivity (2 links)", '#D97706'),
            ("Low connectivity (1 link)", '#0F766E'),
        ]
    ]

    # Build Plotly Figure
    fig = go.Figure(
        data=[edge_trace, node_trace, *legend_traces],
        layout=go.Layout(
            template="plotly_white",  # avoid inheriting a dark default theme
            title=dict(
                text="Transaction Network Topology & Layering Graph",
                font=dict(color="#0F172A", size=16)
            ),
            legend=dict(
                font=dict(color="#0F172A", size=12),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#CBD5E1", borderwidth=1,
                x=1, y=1, xanchor="right", yanchor="top",
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            height=480
        )
    )

    return {"status": "success", "graph_data": fig.to_dict()}
# Add this line at the bottom of backend/graph_engine.py:
generate_transaction_graph = generate_topology_graph


# ============================================================================
# TrustGraph 50% milestone addition (Section 6)
# ----------------------------------------------------------------------------
# generate_topology_graph() above renders customer<->counterparty flow for
# the existing SocGen UI. This is a separate, new calculation: relationship
# RISK based on shared device_id / ip_address with other customers, feeding
# trust_score_tool. "Do not over-engineer" (Section 6) -- kept to counts and
# one weighted formula, no new graph library or persistent graph store.
# ============================================================================

SHARED_DEVICE_WEIGHT = 0.4
SHARED_IP_WEIGHT = 0.3
SUSPICIOUS_LINK_WEIGHT = 0.3
CAP_AT = 3  # counts above this don't add further risk (diminishing returns, avoids one hub node dominating)


def compute_relationship_risk(customer_id: str, df: pd.DataFrame, suspicious_customer_ids: "set | None" = None) -> dict:
    """
    df must contain device_id / ip_address (raw transaction slice, not the
    aggregated feature table). suspicious_customer_ids: optional set of
    customer_ids known to have a ground-truth-confirmed suspicious
    transaction (see evidence_tool.build_historical_case_library) -- used to
    score "connected to a previously suspicious customer" per Section 6.
    """
    if df.empty or "device_id" not in df.columns or "ip_address" not in df.columns:
        return {
            "shared_devices": 0, "shared_ips": 0,
            "connected_suspicious_customers": 0, "graph_risk": 0.0,
            "shared_device_customers": [], "shared_ip_customers": [],
        }

    own_devices = set(df.loc[df.customer_id == customer_id, "device_id"].dropna().unique())
    own_ips = set(df.loc[df.customer_id == customer_id, "ip_address"].dropna().unique())

    device_sharers = set(
        df.loc[df.device_id.isin(own_devices) & (df.customer_id != customer_id), "customer_id"].unique()
    ) if own_devices else set()
    ip_sharers = set(
        df.loc[df.ip_address.isin(own_ips) & (df.customer_id != customer_id), "customer_id"].unique()
    ) if own_ips else set()

    connected_customers = device_sharers | ip_sharers
    suspicious_customer_ids = suspicious_customer_ids or set()
    connected_suspicious = connected_customers & suspicious_customer_ids

    graph_risk = (
        SHARED_DEVICE_WEIGHT * min(len(device_sharers), CAP_AT) / CAP_AT +
        SHARED_IP_WEIGHT * min(len(ip_sharers), CAP_AT) / CAP_AT +
        SUSPICIOUS_LINK_WEIGHT * min(len(connected_suspicious), CAP_AT) / CAP_AT
    )

    return {
        "shared_devices": len(device_sharers),
        "shared_ips": len(ip_sharers),
        "connected_suspicious_customers": len(connected_suspicious),
        "graph_risk": round(float(graph_risk), 4),
        "shared_device_customers": sorted(device_sharers)[:10],
        "shared_ip_customers": sorted(ip_sharers)[:10],
    }