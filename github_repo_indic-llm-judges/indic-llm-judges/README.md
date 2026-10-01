# When Can LLMs Judge Relevance in Indian Languages?

Code, LLM relevance labels and analysis for the paper
**"When Can LLMs Judge Relevance in Indian Languages? Leniency, Calibration and Language Coverage"**
(submitted to ACM Transactions on Information Systems).

Four open-weight LLM judges (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, Aya Expanse 8B) label about 6,000
query–passage pairs from [MIRACL](https://github.com/project-miracl/miracl) in Hindi, Bengali, Telugu,
English, Swahili and Arabic. We compare their labels with MIRACL's native-speaker relevance judgments.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/01_llm_judge_agreement.ipynb` | LLM judging for Hindi, Bengali, Telugu (also used for the truncation ablation) |
| `notebooks/02_control_languages.ipynb` | Same pipeline for English, Swahili, Arabic; full-precision scoring (final runs) |
| `notebooks/03_system_rankings.ipynb` | 13 re-ranking systems scored on every judged pair |
| `notebooks/04_contamination_probe.ipynb` | Query-completion probe for benchmark memorisation |
| `analysis/analysis.py` | All statistics in the paper (bootstrap CIs, calibration, prevalence matching, system rankings) |
| `analysis/figs.py` | Paper figures |
| `results/judgments/` | One file per language × judge × prompt: `qid, docid, human, p_relevant, logodds, passage_tokens, passage_chars` |
| `results/systems/system_scores.csv` | Scores of the 13 systems for every judged pair |
| `results/contamination/contamination_probe.csv` | All 2,280 completion probes |
| `results/tables/` | Final tables and `statistics.json` (all correlations reported in the paper) |

File names follow `{lang}__{model}__{prompt}__t{max_tokens}__clean.csv`; `t512` files are the truncation ablation.

## Reproducing the results

**Analysis only (CPU, a few minutes)**

```bash
pip install -r requirements.txt
python analysis/analysis.py   # writes output/tables/
python analysis/figs.py       # writes output/figures/
```

**Full pipeline (GPU).** The notebooks were run on Kaggle (NVIDIA T4, 16 GB):

1. Import a notebook into Kaggle, enable *GPU T4 x2* and *Internet*.
2. Add a Hugging Face token as the Kaggle secret `HF_TOKEN` and accept the model licences
   (Llama 3.1, Gemma 2, Aya Expanse) on Hugging Face.
3. Set the model and languages in the `CONFIG` cell and run all cells; results are written to `results/`.

All runs use seed 42, deterministic decoding and 4-bit NF4 quantisation.

## Data and licences

Code is released under the MIT licence. Relevance labels, queries and passages come from MIRACL; please
follow the MIRACL licence and cite Zhang et al. (2023). Passage texts are not redistributed here: the
notebooks download them from the official MIRACL release on Hugging Face.

## Citation

```bibtex
@article{TODO2026llmjudges,
  title   = {When Can {LLMs} Judge Relevance in {Indian} Languages? Leniency, Calibration and Language Coverage},
  author  = {TODO},
  journal = {ACM Transactions on Information Systems},
  year    = {2026},
  note    = {Under review}
}
```
