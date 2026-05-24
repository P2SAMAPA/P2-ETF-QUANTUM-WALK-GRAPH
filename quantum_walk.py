import numpy as np
from scipy.linalg import expm

def build_graph_from_returns(returns, weighted=True):
    """
    Build adjacency matrix from ETF returns.
    If weighted: edge weight = max(0, correlation) (or use 1 - distance?).
    We use the absolute correlation as weight (positive only).
    """
    corr = returns.corr().fillna(0)
    if weighted:
        # Use absolute correlation as weight (range 0..1)
        W = np.abs(corr.values)
    else:
        # Unweighted: binary adjacency from threshold (e.g., correlation > 0.5)
        W = (np.abs(corr.values) > 0.5).astype(float)
    np.fill_diagonal(W, 0)
    return W

def continuous_time_quantum_walk(adj, time=1.0):
    """
    Simulate continuous-time quantum walk.
    Hamiltonian = adjacency matrix (weighted).
    Returns probability distribution over nodes after time t.
    """
    n = adj.shape[0]
    # Hamiltonian
    H = adj
    # Unitary evolution operator: U = exp(-i H t)
    U = expm(-1j * H * time)
    # Initial state: uniform superposition over all nodes
    psi0 = np.ones(n, dtype=complex) / np.sqrt(n)
    psi_t = U @ psi0
    prob = np.abs(psi_t)**2
    return prob

def quantum_centrality_scores(returns, time=1.0, weighted=True):
    """
    Compute per-ETF quantum walk probability as centrality score.
    Returns dict: ticker -> probability.
    """
    returns_clean = returns.dropna()
    if returns_clean.shape[1] < 2:
        return {t: 0.0 for t in returns_clean.columns}
    adj = build_graph_from_returns(returns_clean, weighted=weighted)
    prob = continuous_time_quantum_walk(adj, time=time)
    tickers = returns_clean.columns
    return {ticker: prob[i] for i, ticker in enumerate(tickers)}
