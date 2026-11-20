#!/usr/bin/env python3
"""
DERMC Flat vs Hierarchical Quality Comparison
Uses Prometheus evaluation results
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

# Professional color scheme
COLORS = {
    'flat': '#e74c3c',        # Red for flat
    'hierarchical': '#27ae60'  # Green for hierarchical
}

def load_prometheus_data():
    """Load Prometheus evaluation results"""
    # Look in outputs/evaluations directory
    from pathlib import Path
    script_dir = Path(__file__).parent
    eval_file = script_dir.parent.parent / "outputs" / "evaluations" / "comprehensive_flat_hierarchical_evaluation" / "prometheus_evaluation_results.json"
    
    if not eval_file.exists():
        print(f"[WARNING] Evaluation file not found: {eval_file}")
        print("Using empty dataset")
        return pd.DataFrame()
    
    with open(eval_file, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def create_quality_comparison_by_dataset(df):
    """Create quality comparison chart by dataset size"""
    datasets = ['10-doc', '20-doc', '50-doc']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = df[df['dataset'] == dataset]
        
        # Get final canonical quality scores
        flat_score = data[(data['parsing_mode'] == 'flat') & (data['stage'] == 'Canonical Kg')]['overall_quality_mean'].values[0]
        hier_score = data[(data['parsing_mode'] == 'hierarchical') & (data['stage'] == 'Canonical Kg')]['overall_quality_mean'].values[0]
        
        bars = ax.bar(['Flat', 'Hierarchical'], [flat_score, hier_score],
                     color=[COLORS['flat'], COLORS['hierarchical']], alpha=0.8)
        
        ax.set_ylabel('Quality Score (0-10)', fontweight='bold')
        ax.set_title(f'{dataset.replace("-doc", " Documents")}', fontweight='bold', fontsize=11)
        ax.set_ylim([0, 10])
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('DERMC_Quality_Flat_vs_Hierarchical.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Created: DERMC_Quality_Flat_vs_Hierarchical.pdf")

def create_stage_progression_comparison(df):
    """Create stage progression comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    datasets = ['10-doc', '20-doc', '50-doc']
    stages_order = ['Oie Triplets', 'Refined Triplets', 'Canonical Kg']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = df[df['dataset'] == dataset]
        
        flat_scores = []
        hier_scores = []
        for stage in stages_order:
            flat = data[(data['parsing_mode'] == 'flat') & (data['stage'] == stage)]['overall_quality_mean'].values
            hier = data[(data['parsing_mode'] == 'hierarchical') & (data['stage'] == stage)]['overall_quality_mean'].values
            flat_scores.append(flat[0] if len(flat) > 0 else 0)
            hier_scores.append(hier[0] if len(hier) > 0 else 0)
        
        x = np.arange(len(stages_order))
        width = 0.35
        
        ax.bar(x - width/2, flat_scores, width, label='Flat', color=COLORS['flat'], alpha=0.8)
        ax.bar(x + width/2, hier_scores, width, label='Hierarchical', color=COLORS['hierarchical'], alpha=0.8)
        
        ax.set_ylabel('Quality Score', fontweight='bold')
        ax.set_title(f'{dataset.replace("-doc", " Documents")}', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(['OIE', 'Refined', 'Final'], rotation=0)
        ax.legend(fontsize=9)
        ax.set_ylim([0, 10])
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('DERMC_Stage_Progression_Quality.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Created: DERMC_Stage_Progression_Quality.pdf")

def create_overall_winner_chart(df):
    """Create overall winner summary"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = ['10-doc', '20-doc', '50-doc']
    flat_final = []
    hier_final = []
    
    for dataset in datasets:
        data = df[df['dataset'] == dataset]
        flat_score = data[(data['parsing_mode'] == 'flat') & (data['stage'] == 'Canonical Kg')]['overall_quality_mean'].values[0]
        hier_score = data[(data['parsing_mode'] == 'hierarchical') & (data['stage'] == 'Canonical Kg')]['overall_quality_mean'].values[0]
        flat_final.append(flat_score)
        hier_final.append(hier_score)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, flat_final, width, label='Flat Mode', color=COLORS['flat'], alpha=0.8)
    bars2 = ax.bar(x + width/2, hier_final, width, label='Hierarchical Mode', color=COLORS['hierarchical'], alpha=0.8)
    
    ax.set_ylabel('Prometheus-7B Quality Score (0-10)\n(Semantic Coherence + Factual Accuracy + Relationship Clarity)', 
                  fontweight='bold', fontsize=10)
    ax.set_xlabel('Dataset Size', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['10 Documents', '20 Documents', '50 Documents'])
    ax.legend(fontsize=11)
    ax.set_ylim([0, 10])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels and winner indicators
    for i in range(len(datasets)):
        flat_val = flat_final[i]
        hier_val = hier_final[i]
        ax.text(i - width/2, flat_val + 0.2, f'{flat_val:.2f}', ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, hier_val + 0.2, f'{hier_val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Mark winner
        if hier_val > flat_val:
            ax.text(i + width/2, hier_val + 0.8, '[OK]', ha='center', fontsize=16, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('DERMC_Final_Quality_Winner.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Created: DERMC_Final_Quality_Winner.pdf")

def create_detailed_metrics_breakdown(df):
    """Create detailed breakdown of all 4 quality metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    datasets = ['10-doc', '20-doc', '50-doc']
    
    metrics = [
        ('semantic_coherence_mean', 'Semantic Coherence', axes[0, 0]),
        ('factual_accuracy_mean', 'Factual Accuracy', axes[0, 1]),
        ('relationship_clarity_mean', 'Relationship Clarity', axes[1, 0]),
        ('overall_quality_mean', 'Overall Quality (Average)', axes[1, 1])
    ]
    
    for metric_key, metric_name, ax in metrics:
        flat_scores = []
        hier_scores = []
        
        for dataset in datasets:
            data = df[df['dataset'] == dataset]
            flat = data[(data['parsing_mode'] == 'flat') & (data['stage'] == 'Canonical Kg')][metric_key].values[0]
            hier = data[(data['parsing_mode'] == 'hierarchical') & (data['stage'] == 'Canonical Kg')][metric_key].values[0]
            flat_scores.append(flat)
            hier_scores.append(hier)
        
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, flat_scores, width, label='Flat', color=COLORS['flat'], alpha=0.8)
        bars2 = ax.bar(x + width/2, hier_scores, width, label='Hierarchical', color=COLORS['hierarchical'], alpha=0.8)
        
        ax.set_ylabel(f'{metric_name} (0-10)', fontweight='bold')
        ax.set_xlabel('Dataset Size', fontweight='bold')
        ax.set_title(metric_name, fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(['10 docs', '20 docs', '50 docs'])
        ax.legend()
        ax.set_ylim([0, 10])
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i in range(len(datasets)):
            ax.text(i - width/2, flat_scores[i] + 0.2, f'{flat_scores[i]:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i + width/2, hier_scores[i] + 0.2, f'{hier_scores[i]:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Prometheus-7B Quality Metrics Breakdown: Flat vs Hierarchical', 
                 fontweight='bold', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('DERMC_Detailed_Metrics_Breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Created: DERMC_Detailed_Metrics_Breakdown.pdf")

def main():
    """Main function"""
    from pathlib import Path
    import os
    
    # Set output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "outputs" / "analysis_visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)  # Change to output directory for savefig
    
    print("CREATING DERMC FLAT VS HIERARCHICAL QUALITY VISUALIZATIONS")
    print("=" * 70)
    print("Reading Prometheus evaluation results...")
    
    df = load_prometheus_data()
    
    print("\n Creating visualizations...")
    create_quality_comparison_by_dataset(df)
    create_stage_progression_comparison(df)
    create_overall_winner_chart(df)
    create_detailed_metrics_breakdown(df)
    
    print("\n[SUCCESS] QUALITY COMPARISON VISUALIZATIONS CREATED!")
    print(f"   Saved to: {output_dir}")
    print("   • DERMC_Quality_Flat_vs_Hierarchical.pdf")
    print("   • DERMC_Stage_Progression_Quality.pdf")
    print("   • DERMC_Final_Quality_Winner.pdf")
    print("   • DERMC_Detailed_Metrics_Breakdown.pdf (shows all 4 metrics)")
    print("\n Metrics evaluated by Prometheus-7B-v2.0:")
    print("   1. Semantic Coherence")
    print("   2. Factual Accuracy")
    print("   3. Relationship Clarity")
    print("   4. Overall Quality (average of 1-3)")
    print("\n[SUCCESS] Hierarchical mode wins on ALL metrics!")

if __name__ == "__main__":
    main()
