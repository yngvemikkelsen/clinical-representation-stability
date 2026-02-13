"""
Representation Stability Analysis for Clinical Language Models
===============================================================

This script analyzes representation stability across clinical text complexity levels
for 8 transformer models, computing both magnitude and directional metrics.

Models tested:
- GPT-2 Small/Medium/XL (2019)
- BERT Base (2018)
- BioBERT (2020)
- ClinicalBERT (2019)
- Llama-2-7B (2023)
- Mistral-7B (2023)

Metrics computed:
- L2 magnitude: Mean per-token embedding norm
- Cosine similarity: Mean pairwise similarity within texts

Requirements:
- NVIDIA GPU (H100 recommended, A100 works)
- HuggingFace account with access to Llama-2 and Mistral
- ~15 minutes runtime on H100

Usage:
1. Install requirements: pip install -r requirements.txt
2. Set HF_TOKEN environment variable or edit line 50
3. Place all 10 dataset CSV files in same directory
4. Run: python analyze_representations.py
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModel
from huggingface_hub import login
from tqdm import tqdm
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# HuggingFace authentication
HF_TOKEN = os.environ.get('HF_TOKEN', None)
if HF_TOKEN is None:
    raise ValueError(
        "Please set HF_TOKEN environment variable or edit line 50 of this script.\n"
        "Get token from: https://huggingface.co/settings/tokens\n"
        "Accept Llama-2 license: https://huggingface.co/meta-llama/Llama-2-7b-hf\n"
        "Accept Mistral license: https://huggingface.co/mistralai/Mistral-7B-v0.1"
    )

login(token=HF_TOKEN)

# Dataset files (must be in same directory as script)
DATASET_FILES = {
    'g1s': 'mt_samples_simple.csv',
    'g1m': 'mt_samples_moderate.csv',
    'g1c': 'mt_samples_complex.csv',
    'g2s': 'synthetic_simple.csv',
    'g2m': 'synthetic_moderate.csv',
    'g2c': 'synthetic_complex.csv',
    'g3ms': 'lengthcontrol_synth_moderate_simple.csv',
    'g3cs': 'lengthcontrol_synth_complex_simple.csv',
    'g4ms': 'lengthcontrol_real_moderate_simple.csv',
    'g4cs': 'lengthcontrol_real_complex_simple.csv'
}

# Model configurations
MODELS = [
    ('gpt2', 'gpt2', 'gpt'),
    ('gpt2-medium', 'gpt2-medium', 'gpt'),
    ('gpt2-xl', 'gpt2-xl', 'gpt'),
    ('bert', 'bert-base-uncased', 'bert'),
    ('biobert', 'dmis-lab/biobert-v1.1', 'bert'),
    ('clinicalbert', 'emilyalsentzer/Bio_ClinicalBERT', 'bert'),
    ('llama2', 'meta-llama/Llama-2-7b-hf', 'llama'),
    ('mistral', 'mistralai/Mistral-7B-v0.1', 'llama')
]

# Output files
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_datasets():
    """Load all dataset CSV files."""
    datasets = {}
    for key, filename in DATASET_FILES.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dataset file not found: {filename}")
        datasets[key] = pd.read_csv(filename)
    return datasets


def compute_metrics(texts, model, tokenizer, device):
    """
    Compute both L2 magnitude and cosine similarity metrics.
    
    Args:
        texts: List of text strings
        model: Loaded transformer model
        tokenizer: Corresponding tokenizer
        device: torch device (cuda/cpu)
    
    Returns:
        magnitudes: List of mean L2 norms per text
        cosine_sims: List of mean pairwise cosine similarities per text
    """
    magnitudes = []
    cosine_sims = []
    
    for text in tqdm(texts, desc="Processing texts", leave=False):
        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[0] if hasattr(outputs, 'last_hidden_state') else outputs[0][0]
            embeddings_cpu = embeddings.cpu().numpy()
        
        # Compute L2 magnitude
        token_mags = np.linalg.norm(embeddings_cpu, axis=1)
        mean_mag = token_mags.mean()
        magnitudes.append(mean_mag)
        
        # Compute pairwise cosine similarity
        n_tokens = embeddings_cpu.shape[0]
        if n_tokens > 1:
            sims = []
            for i in range(n_tokens):
                for j in range(i+1, n_tokens):
                    sim = 1 - cosine(embeddings_cpu[i], embeddings_cpu[j])
                    sims.append(sim)
            mean_cosine_sim = np.mean(sims)
        else:
            mean_cosine_sim = 1.0
        
        cosine_sims.append(mean_cosine_sim)
    
    return magnitudes, cosine_sims


def load_model(model_id, model_type, device):
    """Load model and tokenizer."""
    if model_type == 'gpt':
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Model.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    elif model_type == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32, device_map='auto')
    else:  # bert
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    
    model.eval()
    return model, tokenizer


def analyze_model(model_name, model_id, model_type, datasets, device):
    """Analyze one model across all datasets."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer = load_model(model_id, model_type, device)
    
    results = []
    for ds_name, df in datasets.items():
        print(f"  Processing {ds_name}...", end=' ')
        texts = df['text'].tolist()
        mags, cosines = compute_metrics(texts, model, tokenizer, device)
        
        result = {
            'model': model_name,
            'dataset': ds_name,
            'mean_magnitude': np.mean(mags),
            'std_magnitude': np.std(mags),
            'mean_cosine_sim': np.mean(cosines),
            'std_cosine_sim': np.std(cosines),
            'magnitudes': mags,
            'cosine_sims': cosines
        }
        results.append(result)
        print(f"✓ Mag: {result['mean_magnitude']:.2f}, Cos: {result['mean_cosine_sim']:.4f}")
    
    # Clean up
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def save_results(all_results):
    """Save detailed and summary results."""
    # Detailed results
    detailed = []
    for r in all_results:
        for i in range(len(r['magnitudes'])):
            detailed.append({
                'model': r['model'],
                'dataset': r['dataset'],
                'text_id': i,
                'magnitude': r['magnitudes'][i],
                'cosine_similarity': r['cosine_sims'][i]
            })
    
    df_detailed = pd.DataFrame(detailed)
    detailed_path = os.path.join(OUTPUT_DIR, 'results_detailed.csv')
    df_detailed.to_csv(detailed_path, index=False)
    print(f"✓ Saved: {detailed_path}")
    
    # Summary results
    summary = []
    for r in all_results:
        summary.append({
            'model': r['model'],
            'dataset': r['dataset'],
            'mean_magnitude': r['mean_magnitude'],
            'std_magnitude': r['std_magnitude'],
            'mean_cosine_sim': r['mean_cosine_sim'],
            'std_cosine_sim': r['std_cosine_sim']
        })
    
    df_summary = pd.DataFrame(summary)
    summary_path = os.path.join(OUTPUT_DIR, 'results_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ Saved: {summary_path}")
    
    return df_summary


def analyze_patterns(df_summary):
    """Analyze and print magnitude vs cosine similarity patterns."""
    print(f"\n{'='*70}")
    print("PATTERN ANALYSIS: Magnitude vs Cosine Similarity")
    print(f"{'='*70}\n")
    
    print("Question: Do magnitude changes accompany directional changes?")
    print("  • Both shift: Geometric effect")
    print("  • Magnitude only: Scaling artifact")
    print("  • Similarity only: Directional change\n")
    
    model_names = df_summary['model'].unique()
    
    for model_name in model_names:
        model_data = df_summary[df_summary['model'] == model_name]
        
        g1s = model_data[model_data['dataset'] == 'g1s']
        g1c = model_data[model_data['dataset'] == 'g1c']
        
        if len(g1s) > 0 and len(g1c) > 0:
            # Magnitude change
            mag_simple = g1s['mean_magnitude'].values[0]
            mag_complex = g1c['mean_magnitude'].values[0]
            mag_change_pct = ((mag_complex - mag_simple) / mag_simple) * 100
            
            # Cosine similarity change
            cos_simple = g1s['mean_cosine_sim'].values[0]
            cos_complex = g1c['mean_cosine_sim'].values[0]
            cos_change_pct = ((cos_complex - cos_simple) / cos_simple) * 100
            
            print(f"{model_name:15s}:")
            print(f"  Magnitude:   {mag_change_pct:+6.2f}%")
            print(f"  Cosine sim:  {cos_change_pct:+6.2f}%")
            
            # Interpretation
            mag_shift = abs(mag_change_pct) > 2.0
            cos_shift = abs(cos_change_pct) > 2.0
            
            if mag_shift and cos_shift:
                print(f"  → Geometric: Both magnitude and direction change")
            elif mag_shift and not cos_shift:
                print(f"  → Scaling: Magnitude changes, direction stable")
            elif not mag_shift and cos_shift:
                print(f"  → Directional: Direction changes, magnitude stable")
            else:
                print(f"  → Stable: Both metrics stable")
            print()


def create_figures(df_summary):
    """Create visualization comparing magnitude vs cosine similarity."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle('Magnitude vs Cosine Similarity Changes (Real Clinical Text)', 
                 fontsize=16, fontweight='bold')
    
    model_names = [m[0] for m in MODELS]
    
    for idx, model_name in enumerate(model_names):
        if idx >= 8:
            break
        
        ax = axes[idx]
        model_data = df_summary[df_summary['model'] == model_name]
        
        # Get simple, moderate, complex for Group 1 (real MT samples)
        g1s_mag = model_data[model_data['dataset'] == 'g1s']['mean_magnitude'].values[0]
        g1m_mag = model_data[model_data['dataset'] == 'g1m']['mean_magnitude'].values[0]
        g1c_mag = model_data[model_data['dataset'] == 'g1c']['mean_magnitude'].values[0]
        
        g1s_cos = model_data[model_data['dataset'] == 'g1s']['mean_cosine_sim'].values[0]
        g1m_cos = model_data[model_data['dataset'] == 'g1m']['mean_cosine_sim'].values[0]
        g1c_cos = model_data[model_data['dataset'] == 'g1c']['mean_cosine_sim'].values[0]
        
        # Normalize to percentage change from simple
        mag_changes = [
            0,
            ((g1m_mag - g1s_mag) / g1s_mag) * 100,
            ((g1c_mag - g1s_mag) / g1s_mag) * 100
        ]
        cos_changes = [
            0,
            ((g1m_cos - g1s_cos) / g1s_cos) * 100,
            ((g1c_cos - g1s_cos) / g1s_cos) * 100
        ]
        
        x = [0, 1, 2]
        ax.plot(x, mag_changes, 'o-', label='Magnitude', linewidth=2, markersize=8, color='blue')
        ax.plot(x, cos_changes, 's--', label='Cosine Sim', linewidth=2, markersize=8, color='red')
        
        ax.set_title(model_name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Simple', 'Moderate', 'Complex'])
        ax.set_ylabel('% Change from Simple')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    figure_path = os.path.join(OUTPUT_DIR, 'magnitude_vs_cosine.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {figure_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis pipeline."""
    print("="*70)
    print("REPRESENTATION STABILITY ANALYSIS")
    print("="*70)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_datasets()
    print(f"✓ Loaded {len(datasets)} datasets")
    
    # Analyze all models
    all_results = []
    for model_name, model_id, model_type in MODELS:
        results = analyze_model(model_name, model_id, model_type, datasets, device)
        all_results.extend(results)
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    df_summary = save_results(all_results)
    
    # Analyze patterns
    analyze_patterns(df_summary)
    
    # Create figures
    print(f"\n{'='*70}")
    print("CREATING FIGURES")
    print(f"{'='*70}")
    create_figures(df_summary)
    
    # Summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Analyzed {len(MODELS)} models across {len(datasets)} datasets")
    print(f"✓ Computed magnitude and cosine similarity metrics")
    print(f"✓ Results saved to {OUTPUT_DIR}/")
    print("\nOutput files:")
    print(f"  • results_detailed.csv - All measurements")
    print(f"  • results_summary.csv - Summary statistics")
    print(f"  • magnitude_vs_cosine.png - Visualization")


if __name__ == '__main__':
    main()
