"""
DERMC Master Execution Script.

Runs complete DERMC experimental pipeline:
1. Run experiments on all test datasets
2. Run TAA evaluation
3. Run Prometheus evaluation
4. Generate visualizations

All outputs saved to experiments/outputs/
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse

# Add parent directory to Python path so dermc package can be imported
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DERMCMasterRunner:
    """Master orchestrator for DERMC experimental pipeline."""

    def __init__(self):
        """Initialize runner with directory structure."""
        self.base_dir = Path(__file__).parent
        self.inputs_dir = self.base_dir / "inputs"
        self.outputs_dir = self.base_dir / "outputs"
        self.scripts_dir = self.base_dir / "scripts"

        # Ensure outputs directory exists
        self.outputs_dir.mkdir(exist_ok=True)
        (self.outputs_dir / "analysis_visuals").mkdir(exist_ok=True)

        logger.info("DERMC Master Execution Script initialized")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Inputs: {self.inputs_dir}")
        logger.info(f"Outputs: {self.outputs_dir}")
        logger.info("=" * 70)
    def _run_script(self, script_path, script_name, cwd=None):
        """
        Run a Python script and handle errors consistently.

        Args:
            script_path: Path to the script file
            script_name: Human-readable name for logging
            cwd: Working directory (default: self.base_dir)

        Returns:
            bool: True if successful, False otherwise
        """
        if not script_path.exists():
            logger.warning(f"[SKIP] {script_name} not found, skipping")
            return False

        logger.info(f" Starting: {script_name}")
        logger.info("")
        
        try:
            # Stream output in real-time
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=cwd or self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Forward subprocess output with indentation
            for line in process.stdout:
                print(f"    {line.rstrip()}")
            
            process.wait()
            logger.info("")
            
            if process.returncode == 0:
                logger.info(f"[OK] COMPLETED: {script_name}")
                return True
            else:
                logger.error(f"[FAILED] FAILED: {script_name} (exit code: {process.returncode})")
                return False
                
        except Exception as e:
            logger.error(f"[FAILED] ERROR: {script_name} - {e}")
            return False

    def run_experiments(self, datasets=None):
        """
        Run DERMC experiments on specified datasets.

        Args:
            datasets: List of dataset names (test10, test20, etc.)
                     None means default: [test10, test20, test50]
        """
        if datasets is None:
            datasets = ['test10', 'test20', 'test50']

        dataset_config = {
            'test10': 'E-Commerce_Documents_test10.csv',
            'test20': 'E-Commerce_Documents_test20.csv',
            'test50': 'E-Commerce_Documents_test50.csv',
            'test200': 'E-Commerce_Documents_test200.csv'
        }

        logger.info("")
        logger.info("=" * 70)
        logger.info("  PHASE 1: DERMC EXPERIMENTS")
        logger.info("=" * 70)
        logger.info(f"Datasets to process: {', '.join(datasets)}")
        logger.info("")

        for idx, dataset in enumerate(datasets, 1):
            if dataset not in dataset_config:
                logger.warning(f"Unknown dataset: {dataset}, skipping")
                continue

            input_file = self.inputs_dir / dataset_config[dataset]
            output_dir = self.outputs_dir / f"enhanced_dermc_output_{dataset}"

            if not input_file.exists():
                logger.error(f"Input file not found: {input_file}")
                continue

            logger.info("-" * 70)
            logger.info(f"[{idx}/{len(datasets)}] Processing dataset: {dataset}")
            logger.info("-" * 70)
            logger.info(f"Input: {input_file.name}")
            logger.info(f"Output: {output_dir.name}")
            logger.info(f"Mode: Hierarchical + Flat (comparison)")
            logger.info(f"LLM Backend: Ollama (llama3.2)")
            logger.info("")
            logger.info("Pipeline Stages:")
            logger.info("  [1/6] Document Parsing (Hierarchical & Flat)")
            logger.info("  [2/6] OIE Extraction")
            logger.info("  [3/6] Coreference Resolution")
            logger.info("  [4/6] Triplet Refinement")
            logger.info("  [5/6] Canonicalization")
            logger.info("  [6/6] Comparison & Export")
            logger.info("")

            # Run DERMC pipeline with Ollama
            # Set PYTHONPATH to include parent directory for dermc imports
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.base_dir.parent) + os.pathsep + env.get('PYTHONPATH', '')
            
            cmd = [
                sys.executable, "-m", "dermc.main",
                "--input", str(input_file),
                "--ontology", str(self.base_dir / "domain_ontology.json"),
                "--output_dir", str(output_dir),
                "--mode", "both",
                "--llm_backend", "ollama",
                "--ollama_model", "llama3.2"
            ]

            try:
                # Stream output in real-time for better visibility
                process = subprocess.Popen(
                    cmd,
                    cwd=self.base_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Stream output line by line
                for line in process.stdout:
                    # Forward subprocess output with indentation
                    print(f"    {line.rstrip()}")
                
                process.wait()
                
                if process.returncode == 0:
                    logger.info("")
                    logger.info(f"[OK] SUCCESS: {dataset} completed")
                else:
                    logger.error(f"[FAILED] FAILED: {dataset} (exit code: {process.returncode})")
            except Exception as e:
                logger.error(f"[FAILED] ERROR: {dataset} - {e}")
            
            logger.info("")
    def run_taa_evaluation(self):
        """Run TAA-inspired evaluation."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PHASE 2: TAA EVALUATION")
        logger.info("=" * 70)
        logger.info("Evaluating knowledge graphs with TAA principles:")
        logger.info("  • Structural plausibility")
        logger.info("  • Compatibility checking")
        logger.info("  • Consistency validation")
        logger.info("  • Semantic plausibility")
        logger.info("")

        script = self.scripts_dir / "evaluation" / "taa_inspired_evaluation.py"
        if not script.exists():
            script = self.scripts_dir / "evaluation" / "true_taa_evaluation.py"

        return self._run_script(script, "TAA Evaluation")

    def run_prometheus_evaluation(self):
        """Run Prometheus evaluation."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PHASE 3: PROMETHEUS EVALUATION")
        logger.info("=" * 70)
        logger.info("Running Prometheus + SHACL validation:")
        logger.info("  • RDF graph validation")
        logger.info("  • SHACL constraint checking")
        logger.info("  • LLM-based quality assessment")
        logger.info("")

        script = (
            self.scripts_dir / "evaluation" / "prometheus_shacl_evaluation.py"
        )
        if not script.exists():
            script = (
                self.scripts_dir / "evaluation" / "quick_prometheus_evaluation.py"
            )

        return self._run_script(script, "Prometheus Evaluation")

    def generate_visualizations(self):
        """Generate visualization PDFs."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PHASE 4: VISUALIZATION GENERATION")
        logger.info("=" * 70)
        logger.info("Generating analysis charts and reports...")
        logger.info("")

        viz_scripts = [
            ("create_quality_visuals.py", "Quality Analysis Charts"),
            ("poster results.py", "Poster Results"),
            ("corrected_dermc_analysis.py", "DERMC Analysis"),
            ("dermc_performance_visualizer.py", "Performance Visualizations")
        ]

        results = []
        for idx, (script_name, display_name) in enumerate(viz_scripts, 1):
            logger.info(f"[{idx}/{len(viz_scripts)}] {display_name}")
            script = self.scripts_dir / "visualization" / script_name
            success = self._run_script(script, display_name)
            results.append(success)
            logger.info("")

        return all(results)
    def run_all(self, datasets=None):
        """
        Run complete pipeline.

        Args:
            datasets: List of datasets to process
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("  DERMC COMPLETE EXPERIMENTAL PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Selected datasets: {', '.join(datasets or ['test10', 'test20', 'test50'])}")
        logger.info("Pipeline phases: Experiments → TAA → Prometheus → Visualizations")
        logger.info("")

        self.run_experiments(datasets)
        self.run_taa_evaluation()
        self.run_prometheus_evaluation()
        self.generate_visualizations()

        logger.info("")
        logger.info("=" * 70)
        logger.info("  [OK] ALL PHASES COMPLETE")
        logger.info("=" * 70)
        logger.info(f"All outputs saved to: {self.outputs_dir}")
        logger.info(
            f"PDFs available in: {self.outputs_dir / 'analysis_visuals'}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='DERMC Master Execution Script',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--experiments-only',
        action='store_true',
        help='Run only experiments (skip evaluations)'
    )
    parser.add_argument(
        '--evaluations-only',
        action='store_true',
        help='Run only evaluations (skip experiments)'
    )
    parser.add_argument(
        '--viz-only',
        action='store_true',
        help='Generate visualizations only'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['test10', 'test20', 'test50', 'test200'],
        default=['test10', 'test20', 'test50'],
        help='Datasets to process (default: test10 test20 test50)'
    )

    args = parser.parse_args()
    runner = DERMCMasterRunner()

    if args.viz_only:
        runner.generate_visualizations()
    elif args.evaluations_only:
        runner.run_taa_evaluation()
        runner.run_prometheus_evaluation()
        runner.generate_visualizations()
    elif args.experiments_only:
        runner.run_experiments(args.datasets)
    else:
        runner.run_all(args.datasets)


if __name__ == "__main__":
    main()
