# Quick Start Guide

Get up and running in 5 minutes.

## Prerequisites

1. **GPU Access**
   - NVIDIA H100 (recommended) or A100
   - Google Colab Pro+ is sufficient

2. **HuggingFace Account**
   - Create account at huggingface.co
   - Get token: Settings → Access Tokens → New Token
   - Accept licenses:
     - https://huggingface.co/meta-llama/Llama-2-7b-hf
     - https://huggingface.co/mistralai/Mistral-7B-v0.1

## Installation (Local)

```bash
# Clone repository
git clone https://github.com/ymikkelsen/clinical-representation-stability.git
cd clinical-representation-stability

# Install dependencies
pip install -r requirements.txt

# Set your HuggingFace token
export HF_TOKEN="hf_your_token_here"

# Run analysis
python analyze_representations.py
```

## Running on Google Colab

1. **Upload files to Colab:**
   - `analyze_representations.py`
   - All 10 CSV files from `data/`

2. **Install requirements:**
```python
!pip install transformers torch pandas numpy scipy matplotlib tqdm huggingface_hub -q
```

3. **Set HuggingFace token:**
```python
import os
os.environ['HF_TOKEN'] = "hf_your_token_here"
```

4. **Run analysis:**
```python
!python analyze_representations.py
```

5. **Download results:**
   - Check `results/` folder
   - Download CSV files and PNG

## Expected Runtime

- **H100:** ~15 minutes
- **A100:** ~25 minutes

## Output Files

You'll get 3 files in `results/`:
1. `results_detailed.csv` - All measurements (~12,000 rows)
2. `results_summary.csv` - Summary statistics (80 rows)
3. `magnitude_vs_cosine.png` - Visualization

## Troubleshooting

**"No such file or directory"**
→ Make sure all 10 CSV files are in the same directory as the script

**"Token not found"**
→ Set HF_TOKEN environment variable or edit line 50 of script

**"CUDA out of memory"**
→ Need H100/A100 GPU (free Colab T4 is insufficient)

**"Access denied to model"**
→ Accept license agreements on HuggingFace for Llama-2 and Mistral

## Next Steps

After getting results:
1. Examine `results_summary.csv` for key findings
2. Check `magnitude_vs_cosine.png` for visual patterns
3. See README.md for detailed documentation
4. Adapt code for your own models/datasets

## Support

Open an issue on GitHub if you encounter problems.
