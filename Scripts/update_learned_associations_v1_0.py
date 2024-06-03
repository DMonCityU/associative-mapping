# Update Learned Associations Script

import sys
import threading
import time
import json
import os
from tqdm import tqdm
from queue import Queue
import datetime
from collections import deque

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
progress_bars_lock = threading.Lock()

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

def create_status_node(session):
    create_status_node_query = f"""
    MERGE (status:OperationStatus {{
        operation: 'UpdateLearnedAssociations'
    }})
    ON CREATE SET status.startTime = timestamp()
    """
    am.run_query(session, create_status_node_query, parameters=None)

def get_total_nodes(session):
    query = """
    MATCH (n1)-[ld:LODC_DEFINITION]->(n2)
    RETURN count(DISTINCT n1) AS total
    """
    result = am.run_query(session, query, parameters=None)
    for record in result:
        total = record['total']
        return total

def get_chunks(session, chunk_size):
    query = """
    MATCH (n1)-[ld:LODC_DEFINITION]->(n2)
    WITH DISTINCT n2 ORDER BY n2.uri
    RETURN collect(n2) AS nodes
    """
    result = am.run_query(session, query, parameters=None)
    for record in result:
        nodes = record['nodes']
        for i in range(0, len(nodes), chunk_size):
            yield nodes[i:i + chunk_size]

def run_chunk(session, chunk):
    for n1 in chunk:
        parameters = {
            "n1Uri": n1['uri']
        }
        query = """
        MATCH (n1)<-[ld1:LODC_DEFINITION]-(n2)-[ld2:LODC_DEFINITION]->(n3)
        WHERE n1 <> n3 AND n1.uri = $n1Uri
        WITH n1, n3, ld1.cosine_similarity AS cs1, ld2.cosine_similarity AS cs2
        WITH n1, n3, cs1 * cs2 AS strength 
        MERGE (n1)-[rel:LEARNED_ASSOCIATION]->(n3) 
        ON CREATE SET rel.strength = strength, rel.createdCount = coalesce(rel.createdCount, 0) + 1 
        ON MATCH SET rel.strength = CASE 
            WHEN rel.strength IS NULL THEN strength 
            ELSE CASE 
                WHEN rel.strength > strength THEN rel.strength 
                ELSE strength 
            END 
        END,
        rel.updatedCount = coalesce(rel.updatedCount, 0) + 1
        WITH count(CASE WHEN rel.createdCount = 1 THEN 1 ELSE NULL END) AS createdRels, 
            count(CASE WHEN rel.updatedCount > 1 THEN 1 ELSE NULL END) AS updatedRels
        MATCH (status:OperationStatus {operation: 'UpdateLearnedAssociations'})
        SET status.lastBatchSize = createdRels + updatedRels,
            status.createdRels = coalesce(status.createdRels, 0) + createdRels,
            status.updatedRels = coalesce(status.updatedRels, 0) + updatedRels,
            status.lastUpdated = timestamp(),
            status.totalProcessed = coalesce(status.totalProcessed, 0) + createdRels + updatedRels,
            status.operations = coalesce(status.operations, 0) + createdRels + updatedRels,
            status.total = coalesce(status.total, 0) + createdRels + updatedRels
        """
        try:
            am.run_query(session, query, parameters)
        except Exception as e:
            tqdm.write(f"Error running chunk for node {n1['uri']}: {e}")
            return False
    return True

def print_status(progress_bars):
    status_query = """
    MATCH (status:OperationStatus {operation: 'UpdateLearnedAssociations'})
    RETURN status.totalProcessed AS totalProcessed, MAX(status.lastUpdated) AS lastUpdated, 
           status.createdRels AS createdRels, status.updatedRels AS updatedRels
    """
    global max_chunks_per_second_5s, chunks_processed_last_interval, interval_start_time

    while True:
        time.sleep(1)
        try:
            with driver.session(database=database) as session:
                status_result = am.run_query(session, status_query, parameters=None)
                for record in status_result:
                    total_processed = record['totalProcessed']
                    created_rels = record['createdRels']
                    updated_rels = record['updatedRels']

                    current_time = time.time()
                    if current_time - interval_start_time >= 5:
                        chunks_processed_this_interval = total_processed - chunks_processed_last_interval
                        chunks_per_second_5s = chunks_processed_this_interval / 5
                        max_chunks_per_second_5s = max(max_chunks_per_second_5s, chunks_per_second_5s)
                        chunks_processed_last_interval = total_processed
                        interval_start_time = current_time

                    with progress_bars_lock:
                        for progress_bar in progress_bars.values():
                            progress_bar.n = total_processed
                            progress_bar.set_postfix({'cr': created_rels, 'ur': updated_rels, 'aw': worker_count})
                            progress_bar.refresh()
        except Exception as e:
            tqdm.write(f"Error fetching status: {e}")

def worker(chunk, progress_bar, state):
    session = driver.session(database=database)
    global worker_count, consecutive_timeouts
    try:
        if not run_chunk(session, chunk):
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
                return
    finally:
        session.close()

def main():
    state = load_state()
    threads = []
    progress_bars = {}

    initial_workers = 1
    max_workers = 10
    worker_count = initial_workers

    created_rels = 0
    updated_rels = 0
    total_processed = 0

    start_time = time.time()

    with driver.session(database=database) as main_session:
        delete_existing_status_nodes(main_session)
        create_status_node(main_session)
        total_nodes = get_total_nodes(main_session)  # Calculate total nodes
        total_chunks = (total_nodes * (total_nodes - 1)) // 2 // chunk_size  # Calculate total chunks
        progress_bar = tqdm(total=total_chunks, desc='UpdateLearnedAssociations', position=0)
        
        # Create and start the status thread
        status_thread = threading.Thread(target=print_status, args=(progress_bars,))
        status_thread.start()

        # Assign chunks to workers
        try:
            for chunk in get_chunks(main_session, chunk_size):
                if chunk in state.get('errored_chunks', []):
                    continue  # Skip errored chunks
                
                # Ensure no more than max_workers threads are active
                while len(threads) >= worker_count:
                    for thread in threads:
                        if not thread.is_alive():
                            threads.remove(thread)
                
                thread = threading.Thread(target=worker, args=(chunk, progress_bar, state))
                thread.start()
                threads.append(thread)
                with progress_bars_lock:
                    progress_bars[id(chunk)] = progress_bar

                # Increment worker count up to max_workers
                if worker_count < max_workers:
                    worker_count += 1

        except (KeyboardInterrupt, Exception) as e:
            tqdm.write(f"Error or interruption received: {e}. Terminating workers...")
            for thread in threads:
                thread.join()
            save_state(state)
            raise

    # Outside the try block, ensuring normal termination
    for thread in threads:
        thread.join()
    progress_bar.close()
    status_thread.join()

    print("All threads completed successfully.")

    # Fetch final status
    with driver.session(database=database) as session:
        status_result = am.run_query(session, """
        MATCH (status:OperationStatus {operation: 'UpdateLearnedAssociations'})
        RETURN status.totalProcessed AS totalProcessed, status.createdRels AS createdRels, status.updatedRels AS updatedRels
        """, parameters=None)
        for record in status_result:
            total_processed = record['totalProcessed']
            created_rels = record['createdRels']
            updated_rels = record['updatedRels']

    total_time = time.time() - start_time
    average_chunks_per_second = total_chunks / total_time

    tqdm.write(f"Summary:\n"
            f"Created relations: {created_rels}\n"
            f"Updated relations: {updated_rels}\n"
            f"Total processed: {total_processed}\n"
            f"Highest worker count: {max_workers}\n"
            f"Max chunks per second (5s interval): {max_chunks_per_second_5s:.2f}\n"
            f"Average chunks per second: {average_chunks_per_second:.2f}\n"
            f"Chunks processed: {total_chunks}")

if __name__ == "__main__":
    main()
