### **Table 4: Triple Progression Analysis Ablation Study**

#### **4a: Pipeline Efficiency Analysis (Flat vs Hierarchical)**

| Dataset | Parsing Mode | OIE→Refined | Refined→Canonical | Overall Efficiency | Quality Control Factor | Information Loss |
|---------|-------------|-------------|-------------------|-------------------|----------------------|------------------|
| **10-doc** | Flat | 82.0% reduction | 0.0% loss | 18.0% retention | 5.55x filtering | Controlled |
| **10-doc** | Hierarchical | 83.8% reduction | 0.0% loss | 16.2% retention | 6.17x filtering | Controlled |
| **20-doc** | Flat | 86.6% reduction | 0.0% loss | 13.4% retention | 7.48x filtering | Controlled |
| **20-doc** | Hierarchical | 89.2% reduction | 0.0% loss | 10.8% retention | 9.28x filtering | Controlled |
| **50-doc** | Flat | 85.0% reduction | 0.0% loss | 15.0% retention | 6.66x filtering | Controlled |
| **50-doc** | Hierarchical | 88.6% reduction | 0.0% loss | 11.4% retention | 8.75x filtering | Controlled |

#### **4b: Parsing Mode Efficiency Comparison**

| Dataset Size | More Efficient Mode | Efficiency Advantage | Filtering Difference | Scalability Rating |
|-------------|-------------------|---------------------|---------------------|-------------------|
| **10-doc** | Flat | 1.8% | 0.62x | A |
| **20-doc** | Flat | 2.6% | 1.80x | A |
| **50-doc** | Flat | 3.6% | 2.09x | B+ |