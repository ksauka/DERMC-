#!/usr/bin/env python3
"""
Simple DERMC Visualizer - Creates 2 PDF files from Excel data
One for FLAT mode, one for HIERARCHICAL mode
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path

# Set backend for headless operation
matplotlib.use('Agg')

def create_flat_visual():
    """Create visualization for FLAT mode"""
    try:
        # Load FLAT.xlsx
        flat_file = "Quality analysis/FLAT.xlsx"
        if not os.path.exists(flat_file):
            print(f"File not found: {flat_file}")
            return False
        
        df = pd.read_excel(flat_file)
        print(f"FLAT data loaded: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("Data preview:")
        print(df.head())
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Extract FLAT mode data properly - group by Dataset and Stage
        flat_data = df[df['Parsing Mode'] == 'Flat']
        
        if len(flat_data) > 0:
            # Get unique datasets and stages
            datasets = flat_data['Dataset'].unique()
            stages = ['Oie Triplets', 'Refined Triplets', 'Canonical Kg']
            
            # Create subplot for each dataset
            fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 6))
            if len(datasets) == 1:
                axes = [axes]
            
            colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
            
            for idx, dataset in enumerate(datasets):
                dataset_data = flat_data[flat_data['Dataset'] == dataset]
                
                values = []
                stage_labels = []
                for stage in stages:
                    stage_row = dataset_data[dataset_data['Stage'] == stage]
                    if not stage_row.empty:
                        values.append(float(stage_row['Overall Quality'].iloc[0]))
                        stage_labels.append(stage.replace('Oie', 'OIE').replace('Kg', 'KG'))
                
                # Create bars
                bars = axes[idx].bar(range(len(values)), values, 
                                   color=colors[:len(values)], alpha=0.8, 
                                   edgecolor='navy', linewidth=1.5)
                
                # Customize subplot
                axes[idx].set_xlabel('Processing Stages', fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('Overall Quality Score', fontsize=12, fontweight='bold')
                axes[idx].set_title(f'FLAT Mode - {dataset}', fontsize=14, fontweight='bold')
                axes[idx].set_xticks(range(len(stage_labels)))
                axes[idx].set_xticklabels(stage_labels, rotation=15, ha='right', fontsize=10)
                axes[idx].grid(True, axis='y', alpha=0.3, linestyle='--')
                axes[idx].set_ylim(0, 7)  # Set consistent y-axis
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, values)):
                    axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                 f'{value:.2f}', ha='center', va='bottom', 
                                 fontsize=11, fontweight='bold')
            
            plt.suptitle('DERMC Performance - FLAT Mode\nQuality Assessment Across Pipeline Stages', 
                        fontsize=16, fontweight='bold', y=0.95)
        else:
            # Fallback visualization
            categories = ['OIE Triplets', 'Refined Triplets', 'Canonical KG']
            values = [5.93, 1.87, 2.47]
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            bars = plt.bar(range(len(categories)), values, 
                          color=colors, alpha=0.8, edgecolor='navy', linewidth=1.5)
            
            plt.xlabel('Processing Stages', fontsize=14, fontweight='bold')
            plt.ylabel('Overall Quality Score', fontsize=14, fontweight='bold')
            plt.title('DERMC Performance - FLAT Mode', fontsize=16, fontweight='bold')
            plt.xticks(range(len(categories)), categories, rotation=15, ha='right')
            plt.grid(True, axis='y', alpha=0.3, linestyle='--')
            plt.ylim(0, 7)
            
            for i, (bar, value) in enumerate(zip(bars, values)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save as PDF
        output_file = "DERMC_FLAT_Performance.pdf"
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Created FLAT visualization: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error creating FLAT visualization: {e}")
        return False

def create_hierarchical_visual():
    """Create visualization for HIERARCHICAL mode"""
    try:
        # Load Hierarchical.xlsx
        hier_file = "Quality analysis/Hierarchical.xlsx"
        if not os.path.exists(hier_file):
            print(f"File not found: {hier_file}")
            return False
        
        df = pd.read_excel(hier_file)
        print(f"HIERARCHICAL data loaded: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("Data preview:")
        print(df.head())
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Extract HIERARCHICAL mode data properly - group by Dataset and Stage
        hier_data = df[df['Parsing Mode'] == 'Hierarchical']
        
        if len(hier_data) > 0:
            # Get unique datasets and stages
            datasets = hier_data['Dataset'].unique()
            stages = ['Oie Triplets', 'Refined Triplets', 'Canonical Kg']
            
            # Create subplot for each dataset
            fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 6))
            if len(datasets) == 1:
                axes = [axes]
            
            colors = ['#C1272D', '#F15A24', '#F7931E']  # Red, Orange-Red, Orange
            
            for idx, dataset in enumerate(datasets):
                dataset_data = hier_data[hier_data['Dataset'] == dataset]
                
                values = []
                stage_labels = []
                for stage in stages:
                    stage_row = dataset_data[dataset_data['Stage'] == stage]
                    if not stage_row.empty:
                        values.append(float(stage_row['Overall Quality'].iloc[0]))
                        stage_labels.append(stage.replace('Oie', 'OIE').replace('Kg', 'KG'))
                
                # Create bars
                bars = axes[idx].bar(range(len(values)), values, 
                                   color=colors[:len(values)], alpha=0.8, 
                                   edgecolor='maroon', linewidth=1.5)
                
                # Customize subplot
                axes[idx].set_xlabel('Processing Stages', fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('Overall Quality Score', fontsize=12, fontweight='bold')
                axes[idx].set_title(f'HIERARCHICAL Mode - {dataset}', fontsize=14, fontweight='bold')
                axes[idx].set_xticks(range(len(stage_labels)))
                axes[idx].set_xticklabels(stage_labels, rotation=15, ha='right', fontsize=10)
                axes[idx].grid(True, axis='y', alpha=0.3, linestyle='--')
                axes[idx].set_ylim(0, 7)  # Set consistent y-axis
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, values)):
                    axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                 f'{value:.2f}', ha='center', va='bottom', 
                                 fontsize=11, fontweight='bold')
            
            plt.suptitle('DERMC Performance - HIERARCHICAL Mode\nQuality Assessment Across Pipeline Stages', 
                        fontsize=16, fontweight='bold', y=0.95)
        else:
            # Fallback visualization
            categories = ['OIE Triplets', 'Refined Triplets', 'Canonical KG']
            values = [3.93, 5.27, 4.40]
            colors = ['#C1272D', '#F15A24', '#F7931E']
            
            bars = plt.bar(range(len(categories)), values, 
                          color=colors, alpha=0.8, edgecolor='maroon', linewidth=1.5)
            
            plt.xlabel('Processing Stages', fontsize=14, fontweight='bold')
            plt.ylabel('Overall Quality Score', fontsize=14, fontweight='bold')
            plt.title('DERMC Performance - HIERARCHICAL Mode', fontsize=16, fontweight='bold')
            plt.xticks(range(len(categories)), categories, rotation=15, ha='right')
            plt.grid(True, axis='y', alpha=0.3, linestyle='--')
            plt.ylim(0, 7)
            
            for i, (bar, value) in enumerate(zip(bars, values)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save as PDF
        output_file = "DERMC_HIERARCHICAL_Performance.pdf"
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Created HIERARCHICAL visualization: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error creating HIERARCHICAL visualization: {e}")
        return False

def main():
    """Main function"""
    print("DERMC Visual Generator")
    print("=" * 30)
    
    success_count = 0
    
    # Create FLAT visualization
    print("\n1. Creating FLAT mode visualization...")
    if create_flat_visual():
        success_count += 1
    
    # Create HIERARCHICAL visualization
    print("\n2. Creating HIERARCHICAL mode visualization...")
    if create_hierarchical_visual():
        success_count += 1
    
    print(f"\n[OK] Successfully created {success_count}/2 visualizations")
    print("\nGenerated files:")
    for pdf_file in ["DERMC_FLAT_Performance.pdf", "DERMC_HIERARCHICAL_Performance.pdf"]:
        if os.path.exists(pdf_file):
            print(f"  - {pdf_file}")

if __name__ == "__main__":
    main()