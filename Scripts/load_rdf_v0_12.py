import sys
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import psutil
import signal

base_path = os.getcwd()

module_path = os.path.join(base_path, "mapping")

# Config
batch_suffix = 'better_querying_test'

data_path = os.path.join(base_path, "data")
batch_path = os.path.join(data_path, batch_suffix if batch_suffix else 'sample')
text_dir = os.path.join(batch_path, 'extracted_prose')
prose_csv_dir = os.path.join(batch_path, 'keyphrase_extraction')
lodc_data_path = os.path.join(data_path, '')
processed_path = os.path.join(batch_path, 'processed')
prose_csv_path = os.path.join(prose_csv_dir, 'combined.csv')
window = 50
database = 'rmf2lod2stig'
chunk_size = 1  # Define a global chunk size variable

node_labels_and_keys = {
    'NormalizedProperty': {
        'fields': ['propertyValue'],
        'identifier': 'uuid',
        'identifier_type': 'plain'
    }
}

# Add the module directory to sys.path if not already included
if module_path not in sys.path:
    sys.path.append(module_path)

from mapping import import_specific_version  # type: ignore

# Import the specific version of the 'all' module
am = import_specific_version('all', '>= 1_9')

from transformers import AutoModel, AutoTokenizer  # type: ignore
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

endpoints = am.load_json(os.path.join(base_path, "config", "endpoints.json"))
terms = am.load_terms()

driver = am.get_neo4j_connection(use_env=True)

def process_nodes(session, nodes_list, pbar):
    start_time = time.time()
    nodes_processed, rdf_queries, merge_queries = am.process_nodes(nodes_list, node_labels_and_keys, terms, tokenizer, model, endpoints, 
                          lodc_data_path, window=window, session=session, max_nodes=1, dry_run=False, similarity_threshold=0.16)
    end_time = time.time()
    elapsed_time = end_time - start_time
    pbar.update(len(nodes_list))
    return nodes_processed, elapsed_time, rdf_queries + merge_queries

def process_nodes_with_retry(driver, nodes_chunk, pbar, max_retries=3):
    for attempt in range(max_retries):
        try:
            with driver.session(database=database) as session:
                return process_nodes(session, nodes_chunk, pbar)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed with error: {e}. Retrying...")
                time.sleep(2)  # Optional: wait a bit before retrying
            else:
                raise

def main(workers):
    # Run the initial query to determine workload
    query = """
    MATCH (n:NormalizedProperty)
    WHERE NOT (n)-[:LODC_DEFINITION]-()
    AND NOT (:Finding)-[:HAS_NORMALIZED_PROPERTY]->(n)
    RETURN n
    ORDER BY rand()
    """

    nodes_list = am.fetch_nodes_wrapper(driver, node_labels_and_keys, database=database, override_query=query)
    remaining_tasks = len(nodes_list)

    if remaining_tasks == 0:
        print("No tasks to process.")
        return

    with tqdm(total=remaining_tasks, desc='Processing nodes', unit='node') as pbar:
        sessions = [driver.session(database=database) for _ in range(workers)]
        futures = []
        result_chunks = [nodes_list[i:i + chunk_size] for i in range(0, len(nodes_list), chunk_size)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while remaining_tasks > 0:
                for session in sessions:
                    if remaining_tasks > 0 and result_chunks:
                        nodes_chunk = result_chunks.pop(0)
                        futures.append(executor.submit(process_nodes_with_retry, driver, nodes_chunk, pbar))
                        remaining_tasks -= len(nodes_chunk)
                        if remaining_tasks <= 0:
                            break

            total_time = 0
            total_nodes = 0
            total_queries = 0

            for future in as_completed(futures):
                result = future.result()
                total_time += result[1]
                total_nodes += result[0]
                total_queries += result[2]

            pbar.set_postfix({
                'cpu': f'{psutil.cpu_percent(interval=1):.1f}%',
                'memory': f'{psutil.virtual_memory().percent:.1f}%'
            })

            average_time_per_worker = total_time / len(futures)
            overall_seconds_per_node = total_time / total_nodes
            total_queries_per_second = total_queries / total_time
            pbar.set_postfix({
                'Avg time/worker': f'{average_time_per_worker:.2f}s',
                'Sec/node': f'{overall_seconds_per_node:.2f}',
                'Queries/sec': f'{total_queries_per_second:.2f}',
                'cpu': f'{psutil.cpu_percent(interval=1):.1f}%',
                'memory': f'{psutil.virtual_memory().percent:.1f}%'
            })

        for session in sessions:
            session.close()

def signal_handler(sig, frame):
    print("\nInterrupt received, stopping...")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neo4J interactor for RDF population.')
    parser.add_argument('--workers', type=int, default=256, help='Number of worker threads')

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    main(args.workers)
