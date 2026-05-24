# Quantum Walk Graph for ETFs

Continuous‑time quantum walk on the ETF correlation graph. The quantum walk yields a probability distribution over ETFs, used as a **quantum centrality** signal. Quantum walks exhibit quadratic speedup in mixing compared to classical random walks, potentially exposing hidden structural relationships.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63, 252, 504, 1008, 2016, 4032, 4536 days)
- Weighted or unweighted graph (correlation strength)
- Continuous‑time quantum walk with Hamiltonian = adjacency matrix
- Unitary evolution via matrix exponential
- Per‑ETF score = probability amplitude after fixed time
- Best window automatically selected (largest absolute raw signal)
- Two‑tab Streamlit dashboard (auto best + manual window selection)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-quantum-walk-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- The quantum walk distributes probability across the graph faster than classical diffusion.
- ETFs that acquire higher probability are those that are structurally more central in a quantum sense – may indicate unique market influence or alpha.
- Complements classical centrality measures (PageRank, eigenvector) with quantum interference effects.

## Requirements

See `requirements.txt`.
