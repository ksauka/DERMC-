"""
DERMC CLI entry point that provides:
1. Command-line interface for pipeline execution
2. LLM backend configuration (OpenAI/Ollama)
3. Processing mode selection (hierarchical/flat/both)
4. Testing and validation utilities

Usage example:
    python -m dermc.main --mode hierarchical --single_mode_only --output_dir my_output
"""
import argparse
import os
import sys
import traceback
import json
import pandas as pd
from dermc.hierarchy_aware_erdc import HierarchyAwareERDC
from dermc.llm_client import LLMClient
from typing import Optional
from dermc.dermc_comprehensive_logger import DERMCLogger
from dermc.coreference_resolver import CoreferenceResolver
import logging

logger = logging.getLogger("dermc.main")
logging.basicConfig(level=logging.INFO)

def main():
    """Main function to run the enhanced DERMC pipeline with optional comparison"""
    # Parse CLI arguments for pipeline configuration
    parser = argparse.ArgumentParser(description='DERMC+ Enhanced Pipeline')
    parser.add_argument('--input', default='E-Commerce_Documents.csv', help='Input CSV file')
    parser.add_argument('--ontology', default='domain_ontology.json', help='Ontology file')
    parser.add_argument('--output_dir', default='enhanced_dermc_output', help='Output directory')
    parser.add_argument('--mode', default='both', choices=['hierarchical', 'flat', 'both'], help='Processing mode (both runs comparison by default)')
    parser.add_argument('--single_mode_only', action='store_true', help='Run only single mode (use with --mode hierarchical/flat)')
    parser.add_argument('--test', action='store_true', help='Run multilingual coreference tests')
    parser.add_argument('--test_pipeline', action='store_true', help='Test full pipeline with sample data')
    parser.add_argument('--llm_backend', default='openai', choices=['openai', 'ollama'], help='LLM backend to use (default: openai)')
    parser.add_argument('--openai_model', default='gpt-3.5-turbo', help='OpenAI model to use (default: gpt-3.5-turbo)')
    parser.add_argument('--ollama_model', default='llama3.2', help='Ollama model to use (default: llama3.2)')
    parser.add_argument('--ollama_url', default='http://localhost:11434/api/chat', help='Ollama API URL (default: http://localhost:11434/api/chat)')
    args = parser.parse_args()

    # Initialize LLM client based on selected backend (Ollama for local, OpenAI for API)
    if args.llm_backend == 'ollama':
        logger.info(f"Initializing Ollama backend with model: {args.ollama_model}")
        llm = LLMClient(
            backend='ollama',
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url
        )
    else:
        logger.info(f"Initializing OpenAI backend with model: {args.openai_model}")
        llm = LLMClient(
            backend='openai',
            model=args.openai_model
        )

    if args.test:
        success = test_multilingual_coreference()
        if not success:
            logger.error("Multilingual testing failed!")
            return

    if args.test_pipeline:
        success = test_pipeline_with_real_data(llm)
        if not success:
            logger.error("Pipeline testing failed!")
            return

    if args.test or args.test_pipeline:
        logger.info("Testing completed successfully!")
        return

    try:
        ontology_path = args.ontology
        if not os.path.exists(ontology_path):
            logger.error(f"Ontology file {ontology_path} not found!")
            return
        input_file = args.input
        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} not found!")
            return
        run_both_modes = (args.mode == 'both')
        single_mode_only = args.single_mode_only
        selected_mode = args.mode
        output_base_dir = args.output_dir
        if single_mode_only and selected_mode in ['hierarchical', 'flat']:
            enable_hierarchical = (selected_mode == "hierarchical")
            comprehensive_logger = DERMCLogger(
                log_file=f"{output_base_dir}/dermc_comprehensive_log.json",
                enable_schema_logging=True,
                enable_coref_logging=True,
                enable_pipeline_logging=True
            )
            pipeline = HierarchyAwareERDC(
                llm=llm, 
                ontology_path=ontology_path,
                enable_hierarchical_parsing=enable_hierarchical,
                comprehensive_logger=comprehensive_logger
            )
            mode_name = "hierarchical" if enable_hierarchical else "flat"
            logger.info(f"Starting Enhanced DERMC pipeline in {mode_name} mode...")
            results_df = pipeline.process_csv(input_file)
            logger.info("Pipeline completed successfully!")
            os.makedirs(output_base_dir, exist_ok=True)
            results_df.to_csv(f"{output_base_dir}/dermc_results_{mode_name}.csv", index=False)
            logger.info(f"Saved main results to {output_base_dir}/dermc_results_{mode_name}.csv")
            oie_data = []
            refined_data = []
            canonical_data = []
            for _, row in results_df.iterrows():
                for item in row.get('oie_triplets', []):
                    if isinstance(item, dict) and 'triplet' in item:
                        s, r, o = item['triplet']
                        oie_data.append({
                            'Doc ID': row.get('Doc ID', ''),
                            'Answer ID': row.get('Answer ID', ''),
                            'Subject': s,
                            'Relation': r,
                            'Object': o,
                            'Hierarchy': item.get('hierarchy', '')
                        })
                for item in row.get('refined_triplets', []):
                    if isinstance(item, dict) and 'triplet' in item:
                        s, r, o = item['triplet']
                        refined_data.append({
                            'Doc ID': row.get('Doc ID', ''),
                            'Answer ID': row.get('Answer ID', ''),
                            'Subject': s,
                            'Relation': r,
                            'Object': o,
                            'Hierarchy': item.get('hierarchy', '')
                        })
                for triplet in row.get('canonical_triplets', []):
                    canonical_data.append({
                        'Doc ID': row.get('Doc ID', ''),
                        'Answer ID': row.get('Answer ID', ''),
                        'Subject': triplet.get('subject', ''),
                        'Subject Type': triplet.get('subject_type', ''),
                        'Relation': triplet.get('relation', ''),
                        'Object': triplet.get('object', ''),
                        'Object Type': triplet.get('object_type', ''),
                        'Hierarchy': triplet.get('hierarchy', ''),
                        'Ontology Version': row.get('Ontology Version', '')
                    })
            if oie_data:
                pd.DataFrame(oie_data).to_csv(f"{output_base_dir}/oie_triplets.csv", index=False)
                logger.info(f"Saved {len(oie_data)} OIE triplets")
            if refined_data:
                pd.DataFrame(refined_data).to_csv(f"{output_base_dir}/refined_triplets.csv", index=False)
                logger.info(f"Saved {len(refined_data)} refined triplets")
            if canonical_data:
                pd.DataFrame(canonical_data).to_csv(f"{output_base_dir}/canonical_kg.csv", index=False)
                logger.info(f"Saved {len(canonical_data)} canonical triplets")
            total_oie = len(oie_data)
            total_refined = len(refined_data) 
            total_canonical = len(canonical_data)
            summary = {
                "pipeline_version": f"enhanced_with_redundancy_reduction_{mode_name}",
                "total_documents_processed": len(results_df),
                "total_oie_triplets": total_oie,
                "total_refined_triplets": total_refined,
                "total_canonical_triplets": total_canonical,
                "redundancy_reduction": {
                    "oie_to_refined_reduction": total_oie - total_refined if total_oie > 0 else 0,
                    "refined_to_canonical_reduction": total_refined - total_canonical if total_refined > 0 else 0
                },
                "enhancement_features": [
                    "semantic_similarity_based_merging",
                    "aggressive_quality_filtering", 
                    "entity_enhancement",
                    "relation_canonicalization"
                ]
            }
            with open(f"{output_base_dir}/dermc_metrics.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Saved enhancement metrics")
            if canonical_data:
                pd.DataFrame(canonical_data).to_csv(f"{output_base_dir}/knowledge_graph.csv", index=False)
            output_files = [
                f"dermc_results_{mode_name}.csv",
                "dermc_metrics.json", 
                "oie_triplets.csv",
                "refined_triplets.csv",
                "canonical_kg.csv",
                "knowledge_graph.csv"
            ]
            for file in output_files:
                filepath = os.path.join(output_base_dir, file)
                if os.path.exists(filepath):
                    logger.info(f"[OK] Generated: {filepath}")
                else:
                    logger.warning(f"[FAILED] Missing: {filepath}")
            comprehensive_logger.finalize_session()
            logger.info(f"Comprehensive log saved to: {output_base_dir}/dermc_comprehensive_log.json")
            print(comprehensive_logger.get_summary_report())
        else:
            logger.info("Starting Enhanced DERMC comparison experiment (hierarchical vs flat parsing)...")
            results = run_both_parsing_modes_enhanced(
                csv_path=input_file,
                ontology_path=ontology_path,
                llm_client=llm,
                output_dir=output_base_dir
            )
            logger.info("Enhanced DERMC comparison completed!")
            logger.info(f"Results saved in {output_base_dir}/ directory")
            print("\n" + "="*60)
            print("ENHANCED ERDRC PIPELINE COMPARISON SUMMARY")
            print("="*60)
            for mode in ['hierarchical', 'flat']:
                if mode in results:
                    result = results[mode]
                    print(f"\n{mode.upper()} MODE:")
                    print(f"  Processing time: {result['processing_time']:.2f}s")
                    print(f"  Total triplets: {result['total_triplets']}")
                    print(f"  Avg triplets/doc: {result['avg_triplets_per_doc']:.2f}")
            h_triplets = results.get('hierarchical', {}).get('total_triplets', 0)
            f_triplets = results.get('flat', {}).get('total_triplets', 0)
            h_time = results.get('hierarchical', {}).get('processing_time', 0)
            f_time = results.get('flat', {}).get('processing_time', 0)
            print(f"\nCOMPARISON:")
            print(f"  Triplet difference (H-F): {h_triplets - f_triplets}")
            print(f"  Time difference (H-F): {h_time - f_time:.2f}s")
            if h_time > 0:
                print(f"  Speed ratio (F/H): {f_time/h_time:.2f}x")
            print(f"\nResults saved to: {output_base_dir}/")
            return
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error(traceback.format_exc())
        raise

# --- Supporting functions migrated from folks_2026.py ---
def test_multilingual_coreference():
    print("Testing Multilingual CoreferenceResolver")
    print("=" * 50)
    try:
        print("Initializing CoreferenceResolver...")
        resolver = CoreferenceResolver()
        test_cases = [
            {"name": "English Text", "text": "John went to the store. He bought some milk and bread. The cashier asked him for payment.", "language": "en"},
            {"name": "Dutch Text", "text": "Jan ging naar de winkel. Hij kocht wat melk en brood. De kassier vroeg hem om betaling.", "language": "nl"},
            {"name": "Mixed Content", "text": "John went to the store in Amsterdam. Hij kocht wat melk. The cashier was vriendelijk and helped him.", "language": "mixed"},
            {"name": "Business Document (Dutch)", "text": "Het bedrijf heeft zijn kwartaalrapport uitgebracht. Het toonde sterke groei. Investeerders waren tevreden met de resultaten.", "language": "nl"}
        ]
        print(f"Testing {len(test_cases)} test cases...")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']} ({test_case['language']})")
            print(f"Original: {test_case['text']}")
            resolved = resolver.resolve_coreferences(test_case['text'])
            print(f"Resolved: {resolved}")
            if resolved != test_case['text']:
                print("Text was modified by coreference resolution")
                changes = sum(1 for a, b in zip(test_case['text'], resolved) if a != b)
                print(f"Character changes: {changes}")
            else:
                print("No changes made to text")
        print("\nMultilingual testing completed!")
        return True
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()
        return False

def test_pipeline_with_real_data(llm_client: Optional[LLMClient] = None):
    print("\nTesting Full Pipeline with Multilingual Data")
    print("=" * 50)
    try:
        sample_data = pd.DataFrame({
            'Doc ID': [1, 2, 3, 4],
            'Answer ID': [1, 2, 3, 4],
            'Document': [
                "John runs a small business. He sells products online. His customers are very satisfied.",
                "Het bedrijf heeft een nieuwe strategie ontwikkeld. Deze strategie richt zich op duurzaamheid. Het management is optimistisch.",
                "The company operates in multiple countries. They have offices in Amsterdam and Berlin. Der CEO besucht regelmäßig die Niederlassungen.",
                "Customer service is very important. De klanten waarderen onze snelle respons. We strive to help them solve their problems quickly."
            ]
        })
        print(f"Created sample dataset with {len(sample_data)} documents")
        client = llm_client if llm_client is not None else LLMClient()
        backend_name = "Ollama" if getattr(client, 'use_ollama', False) else "OpenAI"
        print(f"Using {backend_name} backend for testing")
        pipeline = HierarchyAwareERDC(
            llm=client,
            ontology_path="domain_ontology.json",
            enable_coref=True,
            enable_hierarchical_parsing=True
        )
        print("Pipeline initialized successfully")
        print(f"Coreference enabled: {pipeline.enable_coref}")
        print(f"Hierarchical parsing: {pipeline.enable_hierarchical_parsing}")
        print("\nProcessing sample document...")
        test_doc = sample_data.iloc[0]['Document']
        print(f"Input: {test_doc}")
        if pipeline.coref_resolver:
            resolved = pipeline.coref_resolver.resolve_coreferences(test_doc)
            print(f"After coreference: {resolved}")
        print("Sample processing completed successfully!")
        return True
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        traceback.print_exc()
        return False

def run_both_parsing_modes_enhanced(csv_path: str, ontology_path: str, llm_client: LLMClient, output_dir: str = "enhanced_erdrc_output"):
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    modes = ['hierarchical', 'flat']
    for mode in modes:
        logger.info(f"Running Enhanced ERDRC pipeline in {mode} mode...")
        enable_hierarchical = (mode == 'hierarchical')
        pipeline = HierarchyAwareERDC(
            llm=llm_client,
            ontology_path=ontology_path,
            enable_hierarchical_parsing=enable_hierarchical,
            schema_log_file=f"{output_dir}/schema_retriever_log_{mode}.json"
        )
        import time
        start_time = time.time()
        result_df = pipeline.process_csv(csv_path)
        processing_time = time.time() - start_time
        output_file = f"{output_dir}/dermc_{mode}_results.json"
        with open(output_file, 'w') as f:
            json.dump(result_df.to_dict('records'), f, indent=2)
        csv_output_file = f"{output_dir}/edc_results_{mode}.csv"
        result_df.to_csv(csv_output_file, index=False)
        oie_data = []
        refined_data = []
        canonical_data = []
        for _, row in result_df.iterrows():
            for item in row.get('oie_triplets', []):
                if isinstance(item, dict) and 'triplet' in item:
                    s, r, o = item['triplet']
                    oie_data.append({
                        'Doc ID': row.get('Doc ID', ''),
                        'Answer ID': row.get('Answer ID', ''),
                        'Subject': s,
                        'Relation': r,
                        'Object': o,
                        'Hierarchy': item.get('hierarchy', '')
                    })
            for item in row.get('refined_triplets', []):
                if isinstance(item, dict) and 'triplet' in item:
                    s, r, o = item['triplet']
                    refined_data.append({
                        'Doc ID': row.get('Doc ID', ''),
                        'Answer ID': row.get('Answer ID', ''),
                        'Subject': s,
                        'Relation': r,
                        'Object': o,
                        'Hierarchy': item.get('hierarchy', '')
                    })
            for triplet in row.get('canonical_triplets', []):
                canonical_data.append({
                    'Doc ID': row.get('Doc ID', ''),
                    'Answer ID': row.get('Answer ID', ''),
                    'Subject': triplet.get('subject', ''),
                    'Subject Type': triplet.get('subject_type', ''),
                    'Relation': triplet.get('relation', ''),
                    'Object': triplet.get('object', ''),
                    'Object Type': triplet.get('object_type', ''),
                    'Hierarchy': triplet.get('hierarchy', ''),
                    'Ontology Version': row.get('Ontology Version', '')
                })
        if oie_data:
            pd.DataFrame(oie_data).to_csv(f"{output_dir}/oie_triplets_{mode}.csv", index=False)
        if refined_data:
            pd.DataFrame(refined_data).to_csv(f"{output_dir}/refined_triplets_{mode}.csv", index=False)
        if canonical_data:
            pd.DataFrame(canonical_data).to_csv(f"{output_dir}/canonical_kg_{mode}.csv", index=False)
            pd.DataFrame(canonical_data).to_csv(f"{output_dir}/knowledge_graph_{mode}.csv", index=False)
        total_oie = len(oie_data)
        total_refined = len(refined_data) 
        total_canonical = len(canonical_data)
        metrics = {
            "pipeline_version": f"dermc_{mode}",
            "total_documents_processed": len(result_df),
            "total_oie_triplets": total_oie,
            "total_refined_triplets": total_refined,
            "total_canonical_triplets": total_canonical,
            "processing_time_seconds": processing_time,
            "triplets_per_document": total_canonical / len(result_df) if len(result_df) > 0 else 0
        }
        with open(f"{output_dir}/edc_metrics_{mode}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        total_triplets = sum(len(row.get('canonical_triplets', [])) for _, row in result_df.iterrows())
        results[mode] = {
            'parsing_mode': mode,
            'processing_time': processing_time,
            'total_documents': len(result_df),
            'total_triplets': total_triplets,
            'avg_triplets_per_doc': total_triplets / len(result_df) if len(result_df) > 0 else 0,
            'output_file': output_file
        }
        logger.info(f"Enhanced ERDRC {mode} mode completed in {processing_time:.2f}s with {total_triplets} triplets")
    comparison_report = generate_enhanced_comparison_report(results)
    report_file = f"{output_dir}/dermc_comparison_report.json"
    with open(report_file, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    logger.info(f"Enhanced ERDRC comparison completed. Report saved to {report_file}")
    return results

def generate_enhanced_comparison_report(results):
    hierarchical = results.get('hierarchical', {})
    flat = results.get('flat', {})
    report = {
        'pipeline': 'ERDRC+ Enhanced',
        'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
        'comparison_summary': {
            'hierarchical_mode': {
                'processing_time_seconds': hierarchical.get('processing_time', 0),
                'total_documents': hierarchical.get('total_documents', 0),
                'total_triplets': hierarchical.get('total_triplets', 0),
                'avg_triplets_per_doc': hierarchical.get('avg_triplets_per_doc', 0)
            },
            'flat_mode': {
                'processing_time_seconds': flat.get('processing_time', 0),
                'total_documents': flat.get('total_documents', 0),
                'total_triplets': flat.get('total_triplets', 0),
                'avg_triplets_per_doc': flat.get('avg_triplets_per_doc', 0)
            }
        },
        'performance_analysis': {
            'triplet_extraction_difference': hierarchical.get('total_triplets', 0) - flat.get('total_triplets', 0),
            'processing_time_difference_seconds': hierarchical.get('processing_time', 0) - flat.get('processing_time', 0),
            'efficiency_ratio': flat.get('processing_time', 1) / hierarchical.get('processing_time', 1) if hierarchical.get('processing_time', 0) > 0 else 0
        },
        'recommendations': {
            'preferred_mode': 'hierarchical' if hierarchical.get('total_triplets', 0) > flat.get('total_triplets', 0) else 'flat',
            'reasoning': 'Hierarchical mode extracts more triplets' if hierarchical.get('total_triplets', 0) > flat.get('total_triplets', 0) else 'Flat mode is more efficient'
        }
    }
    return report

if __name__ == "__main__":
    main()
