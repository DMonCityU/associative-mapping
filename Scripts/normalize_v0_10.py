# Script name: Normalize

import sys
import threading
import time
import json
import os
from tqdm import tqdm
import argparse
from collections import deque
import datetime

print("Importing module")

base_path = os.getcwd()

module_path = os.path.join(base_path, "mapping")

# Add the module directory to sys.path if not already included
if module_path not in sys.path:
    sys.path.append(module_path)

from mapping import import_specific_version  # type: ignore

# Import the specific version of the 'all' module
am = import_specific_version('all', '>= 1_9')

print("Module imported")

database = 'rmf2lod2stig'
state_file_path = 'state.json'
chunk_size = 100  # Define chunk size for internal chunking

driver = am.get_neo4j_connection(use_env=True)

# Initialize timeout count
consecutive_timeouts = 0
timeouts_queue = deque(maxlen=5)

initial_workers = 1
max_workers = 10
worker_count = initial_workers
chunks_processed_last_interval = 0
max_chunks_per_second_5s = 0
interval_start_time = time.time()

node_labels_and_keys = {
    'Finding': ['checktext', 'description', 'fixtext', 'title'],
    'Backmatter':  ['citation', 'title'],
    'ControlParam': ['guidelines_prose', 'label', 'prose'],
    'rev5Control':  ['title'],
    'rev5Group': ['title'],
    'SubPart': ['prose'],
    'ControlPart': ['prose']
}

def load_state():
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError):
            os.remove(state_file_path)
    return {}

def save_state(state):
    # Convert datetime objects to strings before saving state
    for key, value in state.items():
        if isinstance(value, (datetime.datetime, datetime.date)):
            state[key] = value.isoformat()
    with open(state_file_path, 'w') as f:
        json.dump(state, f, indent=4)

def mark_chunk_as_errored(state, chunk):
    if 'errored_chunks' not in state:
        state['errored_chunks'] = []
    state['errored_chunks'].append(chunk)
    save_state(state)

def delete_existing_status_nodes(session):
    delete_query = "MATCH (status:OperationStatus) DELETE status"
    am.run_query(session, delete_query, parameters=None)

def create_status_node(session, node_label, node_property, query_operation='Normalize'):
    create_status_node_query = f"""
    CREATE (status:OperationStatus {{
        operation: '{query_operation}_{node_label}_{node_property}',
        lastBatchSize: 0,
        lastUpdated: timestamp(),
        totalProcessed: 0,
        batches: 0,
        total: 0,
        failedBatches: 0,
        failedOperations: 0,
        batch: 0,
        operations: 0,
        retries: 0,
        errorMessages: []
    }})
    """
    am.run_query(session, create_status_node_query, parameters=None)

def get_total_nodes(session, node_label, node_property):
    query = f"""
    MATCH (n:{node_label}) WHERE n.{node_property} IS NOT NULL AND NOT (n)-[:HAS_NORMALIZED_PROPERTY]->(:NormalizedProperty {{propertyName: '{node_property}'}}) RETURN count(n) AS total
    """
    result = am.run_query(session, query, parameters=None)
    for record in result:
        return record['total']

def get_total_nodes_update_relationships(session, node_label):
    query = f"""
    MATCH (np:NormalizedProperty)<-[:HAS_NORMALIZED_PROPERTY]-(s:{node_label})-[ld:LODC_DEFINITION]->(target)
    WHERE ld.field = np.propertyName
    RETURN count(np) AS total
    """
    result = am.run_query(session, query, parameters=None)
    for record in result:
        return record['total']

def get_chunks(session, node_label, node_property, chunk_size):
    query = f"""
    MATCH (n:{node_label}) WHERE n.{node_property} IS NOT NULL AND NOT (n)-[:HAS_NORMALIZED_PROPERTY]->(:NormalizedProperty {{propertyName: '{node_property}'}})
    RETURN n ORDER BY n.{node_property}
    """
    result = am.run_query(session, query, parameters=None)
    nodes = [record['n'] for record in result]
    for i in range(0, len(nodes), chunk_size):
        yield nodes[i:i + chunk_size]

def run_chunk(session, chunk, node_label, node_property, query_operation='Normalize'):
    for node in chunk:
        parameters = {
            "nodeProperty": node[node_property],
            "nodeLabel": node_label,
            "nodePropertyKey": node_property
        }
        query = f"""
        MATCH (n:{node_label}) WHERE n.{node_property} = $nodeProperty
        MERGE (np1:NormalizedProperty {{propertyName: '{node_property}', propertyValue: $nodeProperty}})
        MERGE (n)-[:HAS_NORMALIZED_PROPERTY]->(np1)
        WITH count(*) AS batchSize
        MATCH (status:OperationStatus {{operation: '{query_operation}_{node_label}_{node_property}'}})
        SET status.lastBatchSize = batchSize, 
            status.lastUpdated = timestamp(), 
            status.totalProcessed = coalesce(status.totalProcessed, 0) + batchSize
        """
        try:
            am.run_query(session, query, parameters)
        except Exception as e:
            tqdm.write(f"Error running chunk for {node[node_property]}: {e}")
            return False
    return True

def run_chunk_update_relationships(session, chunk, node_label):
    for node in chunk:
        parameters = {
            "nodeUri": node['uri']
        }
        query = f"""
        MATCH (np:NormalizedProperty)<-[:HAS_NORMALIZED_PROPERTY]-(s:{node_label})-[ld:LODC_DEFINITION]->(target)
        WHERE ld.field = np.propertyName
        WITH np, s, ld, target
        MERGE (np)-[newLd:LODC_DEFINITION]->(target)
        SET newLd += properties(ld)
        DELETE ld
        WITH count(*) AS batchSize
        MATCH (status:OperationStatus {{operation: 'UpdateRelationships_{node_label}'}})
        SET status.lastBatchSize = batchSize,
            status.lastUpdated = timestamp(),
            status.totalProcessed = coalesce(status.totalProcessed, 0) + batchSize
        """
        try:
            am.run_query(session, query, parameters)
        except Exception as e:
            tqdm.write(f"Error running chunk for {node['uri']}: {e}")
            return False
    return True

def worker(chunk, progress_bar, state, node_label, node_property, operation):
    session = driver.session(database=database)
    global worker_count, consecutive_timeouts
    try:
        if operation == "normalize":
            if not run_chunk(session, chunk, node_label, node_property):
                mark_chunk_as_errored(state, chunk)
        elif operation == "update_relationships":
            if not run_chunk_update_relationships(session, chunk, node_label):
                mark_chunk_as_errored(state, chunk)
        progress_bar.update(len(chunk))
    except Exception as e:
        if "failed to obtain a connection from the pool within 30s" in str(e):
            tqdm.write(f"Connection timeout for chunk. Marking chunk for retry: {chunk}")
            mark_chunk_as_errored(state, chunk)
            worker_count = max(1, worker_count - 1)
            timeouts_queue.append(True)
            if timeouts_queue.count(True) == 5 and worker_count == 1:
                tqdm.write("5 consecutive timeouts with only one worker. Terminating...")
                save_state(state)
                sys.exit(1)
    finally:
        session.close()

def get_chunks_update_relationships(session, node_label, chunk_size):
    query = f"""
    MATCH (np:NormalizedProperty)<-[:HAS_NORMALIZED_PROPERTY]-(s:{node_label})-[ld:LODC_DEFINITION]->(target)
    WHERE ld.field = np.propertyName
    RETURN np, s, ld, target ORDER BY np.uri
    """
    result = am.run_query(session, query, parameters=None)
    nodes = [record['np'] for record in result]
    for i in range(0, len(nodes), chunk_size):
        yield nodes[i:i + chunk_size]

def create_status_node_update_relationships(session, node_label):
    create_status_node_query = f"""
    CREATE (status:OperationStatus {{
        operation: 'UpdateRelationships_{node_label}',
        lastBatchSize: 0,
        lastUpdated: timestamp(),
        totalProcessed: 0,
        batches: 0,
        total: 0,
        failedBatches: 0,
        failedOperations: 0,
        batch: 0,
        operations: 0,
        retries: 0,
        errorMessages: []
    }})
    """
    am.run_query(session, create_status_node_query, parameters=None)

def main():
    state = load_state()
    threads = []
    progress_bars = {}
    global worker_count

    parser = argparse.ArgumentParser(description='Run normalization and relationship update operations.')
    parser.add_argument('--normalize', action='store_true', help='Run the normalization operation.')
    parser.add_argument('--update-relationships', action='store_true', help='Run the update relationships operation.')
    args = parser.parse_args()

    if not args.normalize and not args.update_relationships:
        print("No operations specified. Use --normalize and/or --update-relationships flags to run respective operations.")
        return

    operations = []
    if args.normalize:
        operations.append("normalize")
    if args.update_relationships:
        operations.append("update_relationships")

    with driver.session(database=database) as main_session:
        delete_existing_status_nodes(main_session)

        for operation in operations:
            if operation == "normalize":
                for node_label, properties in node_labels_and_keys.items():
                    for node_property in properties:
                        total_nodes = get_total_nodes(main_session, node_label, node_property)
                        create_status_node(main_session, node_label, node_property)
                        total_chunks = (total_nodes // chunk_size) + 1
                        progress_bar = tqdm(total=total_chunks, desc=f'Normalize_{node_label}_{node_property}', position=len(threads))

                        try:
                            for chunk in get_chunks(main_session, node_label, node_property, chunk_size):
                                if chunk in state.get('errored_chunks', []):
                                    continue  # Skip errored chunks

                                # Ensure no more than max_workers threads are active
                                while len(threads) >= worker_count:
                                    for thread in threads:
                                        if not thread.is_alive():
                                            threads.remove(thread)

                                thread = threading.Thread(target=worker, args=(chunk, progress_bar, state, node_label, node_property, "normalize"))
                                thread.start()
                                threads.append(thread)

                                # Increment worker count up to max_workers
                                if worker_count < max_workers:
                                    worker_count += 1

                        except (KeyboardInterrupt, Exception) as e:
                            tqdm.write(f"Error or interruption received: {e}. Terminating workers...")
                            for thread in threads:
                                thread.join()
                            save_state(state)
                            raise

            elif operation == "update_relationships":
                for node_label in node_labels_and_keys:
                    total_nodes = get_total_nodes_update_relationships(main_session, node_label)
                    total_chunks = (total_nodes // chunk_size) + 1
                    create_status_node_update_relationships(main_session, node_label)
                    progress_bar = tqdm(total=total_chunks, desc=f'UpdateRelationships_{node_label}', position=len(threads))

                    try:
                        for chunk in get_chunks_update_relationships(main_session, node_label, chunk_size):
                            if chunk in state.get('errored_chunks', []):
                                continue  # Skip errored chunks

                            # Ensure no more than max_workers threads are active
                            while len(threads) >= worker_count:
                                for thread in threads:
                                    if not thread.is_alive():
                                        threads.remove(thread)

                            thread = threading.Thread(target=worker, args=(chunk, progress_bar, state, node_label, node_property, "update_relationships"))
                            thread.start()
                            threads.append(thread)

                            # Increment worker count up to max_workers
                            if worker_count < max_workers:
                                worker_count += 1

                    except (KeyboardInterrupt, Exception) as e:
                        tqdm.write(f"Error or interruption received: {e}. Terminating workers...")
                        for thread in threads:
                            thread.join()
                        save_state(state)
                        raise

        for thread in threads:
            thread.join()
        progress_bar.close()

        print("All threads completed successfully.")

if __name__ == "__main__":
    main()
