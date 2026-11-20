### **Table 3: Prometheus-7B-v2.0 LLM Evaluation Ablation Study**

#### **3a: Stage-by-Stage Quality Analysis (Flat vs Hierarchical)**

| Dataset | Parsing Mode | Stage | Semantic Coherence | Factual Accuracy | Relationship Clarity | Overall Quality | Samples | Max Score Achieved |
|---------|-------------|-------|-------------------|------------------|---------------------|-----------------|---------|-------------------|
| **10-doc** | Flat | Canonical Kg | 2.47 ± 3.12 | 2.47 ± 3.12 | 2.47 ± 3.12 | 2.47 ± 3.12 | 15 | 10.00 |
| **10-doc** | Flat | Oie Triplets | 5.93 ± 3.66 | 5.93 ± 3.66 | 5.93 ± 3.66 | 5.93 ± 3.66 | 15 | 10.00 |
| **10-doc** | Flat | Refined Triplets | 1.87 ± 2.39 | 1.87 ± 2.39 | 1.87 ± 2.39 | 1.87 ± 2.39 | 15 | 10.00 |
| **10-doc** | Hierarchical | Canonical Kg | 4.40 ± 4.22 | 4.40 ± 4.22 | 4.40 ± 4.22 | 4.40 ± 4.22 | 15 | 10.00 |
| **10-doc** | Hierarchical | Oie Triplets | 3.93 ± 3.89 | 3.93 ± 3.89 | 3.93 ± 3.89 | 3.93 ± 3.89 | 15 | 10.00 |
| **10-doc** | Hierarchical | Refined Triplets | 5.27 ± 3.89 | 5.27 ± 3.89 | 5.27 ± 3.89 | 5.27 ± 3.89 | 15 | 10.00 |
| **20-doc** | Flat | Canonical Kg | 3.33 ± 3.59 | 3.33 ± 3.59 | 3.33 ± 3.59 | 3.33 ± 3.59 | 15 | 10.00 |
| **20-doc** | Flat | Oie Triplets | 4.27 ± 4.17 | 4.27 ± 4.17 | 4.27 ± 4.17 | 4.27 ± 4.17 | 15 | 10.00 |
| **20-doc** | Flat | Refined Triplets | 3.67 ± 3.94 | 3.67 ± 3.94 | 3.67 ± 3.94 | 3.67 ± 3.94 | 15 | 10.00 |
| **20-doc** | Hierarchical | Canonical Kg | 4.53 ± 3.74 | 4.53 ± 3.74 | 4.53 ± 3.74 | 4.53 ± 3.74 | 15 | 10.00 |
| **20-doc** | Hierarchical | Oie Triplets | 5.47 ± 4.35 | 5.47 ± 4.35 | 5.47 ± 4.35 | 5.47 ± 4.35 | 15 | 10.00 |
| **20-doc** | Hierarchical | Refined Triplets | 4.07 ± 3.96 | 4.07 ± 3.96 | 4.07 ± 3.96 | 4.07 ± 3.96 | 15 | 10.00 |
| **50-doc** | Flat | Canonical Kg | 3.00 ± 3.48 | 3.00 ± 3.48 | 3.00 ± 3.48 | 3.00 ± 3.48 | 15 | 10.00 |
| **50-doc** | Flat | Oie Triplets | 4.27 ± 4.17 | 4.27 ± 4.17 | 4.27 ± 4.17 | 4.27 ± 4.17 | 15 | 10.00 |
| **50-doc** | Flat | Refined Triplets | 3.73 ± 3.64 | 3.73 ± 3.64 | 3.73 ± 3.64 | 3.73 ± 3.64 | 15 | 10.00 |
| **50-doc** | Hierarchical | Canonical Kg | 5.13 ± 4.18 | 5.13 ± 4.18 | 5.13 ± 4.18 | 5.13 ± 4.18 | 15 | 10.00 |
| **50-doc** | Hierarchical | Oie Triplets | 4.53 ± 4.08 | 4.53 ± 4.08 | 4.53 ± 4.08 | 4.53 ± 4.08 | 15 | 10.00 |
| **50-doc** | Hierarchical | Refined Triplets | 3.33 ± 3.59 | 3.33 ± 3.59 | 3.33 ± 3.59 | 3.33 ± 3.59 | 15 | 10.00 |

#### **3b: Parsing Mode Performance Comparison**

| Dataset Size | Best Performing Mode | Quality Difference | Peak Score Difference | Recommendation |
|-------------|---------------------|-------------------|----------------------|----------------|
| **10-doc** | Hierarchical | 1.11 | 0.00 | Clear Winner |
| **20-doc** | Hierarchical | 0.93 | 0.00 | Clear Winner |
| **50-doc** | Hierarchical | 0.67 | 0.00 | Clear Winner |