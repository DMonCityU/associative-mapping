import os
import json
import logging
import csv
import pandas as pd
from functools import wraps
from urllib.parse import unquote
import spacy
import torch
import subprocess
import time
import jmespath
import traceback

override_logger = False # FIXME handle this in a more reasonable way.

functions_without_dependencies = []
functions_to_ignore = []

report_time = True      # Return the time for execution of functions under audit.
LOG_LEVELS = ['DEBUG','INFO','WARNING','ERROR','CRITICAL']

fname = os.path.splitext(os.path.basename(__file__))[0]
abspath = os.path.dirname(os.path.abspath(__file__))
deptree_fname = os.path.join(abspath, f"{fname}_dependency_tree.json")
module_file_path = os.path.abspath(__file__)
dependency_tree_script = os.path.join(abspath, "dependency_tree.py")

try:
    result = subprocess.run(
        f"python {dependency_tree_script} {module_file_path} {deptree_fname}",
        shell=True,
        check=True,
        capture_output=True,
        text=True
    )

except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e.stderr}")
    raise

with open(deptree_fname, 'r') as file:
    function_dependencies = json.loads(file.read())

def collect_dependencies(function_list, functions_to_ignore, function_dependencies):
    def helper(function, depth, dependencies, visited):
        # Avoid functions calling themselves
        if function in visited:
            return
        
        # Suppress ignored functions
        if function in functions_to_ignore: 
            if function not in dependencies:
                dependencies[function] = {'logging_level': "CRITICAL"}
            if dependencies[function] != {'logging_level': "CRITICAL"}:
                dependencies[function] = {'logging_level': "CRITICAL"}
            return

        visited.add(function)
        items = function_dependencies.get(function, {}).get('downstream', [])

        current_level = LOG_LEVELS[min(depth, len(LOG_LEVELS) - 1)]  # Get the appropriate log level based on depth

        # Check if the function is not yet in the dependencies or needs updating
        if function not in dependencies or LOG_LEVELS.index(dependencies[function]['logging_level']) > LOG_LEVELS.index(current_level):
            dependencies[function] = {'logging_level': current_level, 'report_time': True}
        
        for item in items:
            if isinstance(item, str):
                helper(item, depth + 1, dependencies, visited)
            elif isinstance(item, dict):
                for sub_function in item:
                    helper(sub_function, depth + 1, dependencies, visited)

        visited.remove(function)

    all_dependencies = {}

    for function in function_list:
        all_dependencies[function] = {'logging_level': "DEBUG", 'report_time': True}
        helper(function, 0, all_dependencies, set())
    
    return all_dependencies

if functions_without_dependencies:
    functions_to_audit_unfiltered = collect_dependencies(functions_without_dependencies, functions_to_ignore, function_dependencies)
    functions_to_audit = {k: v for k, v in functions_to_audit_unfiltered.items() if k not in functions_to_ignore}
else: functions_to_audit = dict() # No functions

# Map logging level names to numeric values
LOG_LEVELS_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def configure_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s : %(lineno)d : %(funcName)s : %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = configure_logger('annotation_logger')

def audit_logger(functions_to_audit):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            prev_loglevel = logger.getEffectiveLevel()
            if func_name in functions_to_audit and not override_logger:
                logging_level_name = functions_to_audit[func_name]['logging_level']
                logging_level = LOG_LEVELS_MAP[logging_level_name]
                logger.setLevel(logging_level)
                report_time = functions_to_audit[func_name].get('report_time', False)
                if report_time:
                    start_time = time.time()
            else:
                logger.setLevel(logging.CRITICAL)

            result = func(*args, **kwargs)

            if func_name in functions_to_audit and report_time:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.debug(f"{func_name} executed in {execution_time:.4f} seconds")
            logger.setLevel(prev_loglevel)
            return result
        return wrapper
    return decorator

import ahocorasick, csv, glob, hashlib, itertools, inflect
import json, numpy as np
import os, pandas as pd, random, rdflib
import re, requests, spacy, torch, urllib
from tqdm import tqdm
import torch.nn.functional as F
from urllib.parse import unquote
from neo4j import GraphDatabase

nlp = spacy.load("en_core_web_sm")

inflector = inflect.engine()

@audit_logger(functions_to_audit)
def local_path(file_name):
    return os.path.join(os.getcwd(), file_name)

@audit_logger(functions_to_audit)
def encode_contexts(mentions, full_text, tokenizer, model, window=50):
    """
    Encodes multiple text contexts using the provided SBERT model and tokenizer,
    each associated with its term and position information. It uses a sliding window to extract context around each mention
    and then encodes it.

    Args:
        mentions (list of tuples): List where each tuple contains a term and its start and end positions in the full text.
        full_text (str): The complete text from which contexts are derived.
        tokenizer (AutoTokenizer): Tokenizer from the SBERT model used for tokenizing the text.
        model (AutoModel): The pre-trained SBERT model used for generating embeddings.
        window (int, optional): The number of words around the mention to consider for creating the context. Defaults to 5.

    Returns:
        list of tuples: Each tuple contains the term, its embedding, and the start position of the term in the text.
    """
    context_embeddings = []
    seen_mentions = set()  # Set to track processed mentions

    # Ensure mentions is a list of tuples
    if isinstance(mentions, tuple):
        mentions = [mentions]
    elif not all(isinstance(mention, tuple) and len(mention) == 3 for mention in mentions):
        logger.error("Mentions must be a list of tuples with (term, start_pos, end_pos).")
        return context_embeddings


    # Deduplicate mentions
    before_len = len(mentions)
    mentions = list(set(mentions))
    if len(mentions) < before_len:
        logger.warning(f"encode_contexts removed {before_len - len(mentions)} duplicate mentions")
    if len(mentions) == 0:
        # TODO re-enable for full run
        #logger.error("No remaining mentions! Something went wrong.")
        return context_embeddings  # Return early if no mentions are left

    for mention in mentions:
        logger.info(f"Processing mention: {mention}")
        # Convert mention to a hashable form to check for duplicates
        if not mention or len(mention) < 3:
            logger.error(f"Invalid mention format: {mention}")
            continue
        mention_key = (mention[0], mention[1], mention[2])  # Mention is a tuple (term, start_pos, end_pos)

        if mention_key in seen_mentions:
            logger.warning(f"Duplicate mention detected: {mention}")
        else:
            seen_mentions.add(mention_key)
            logger.debug(f"Mention: {mention}")
            context = get_context(full_text, mention, window)
            logger.debug(f"Context: {context}")
            # Call encode_context with the context text and mention details
            encoded_context = encode_context(mention, context, tokenizer, model)
            # Store the tuple without the end_pos, as it is redundant
            context_embeddings.append(encoded_context)

    return context_embeddings

@audit_logger(functions_to_audit)
def process_multiple_samples(sample_texts, mentions_list, tokenizer, model, window=5, 
                             output_dir=None,start_index=None):
    """
    Process multiple text samples with their corresponding mentions using the encode_contexts function.
    Optionally saves the encoded results and the sample texts to a directory.
    
    Args:
        sample_texts (list of str): A list of text samples from which contexts are derived.
        mentions_list (list of list of tuples): Each list contains tuples for each text, where each tuple
                                                contains a term and its start and end positions.
        tokenizer (AutoTokenizer): Tokenizer from the SBERT model used for tokenizing the text.
        model (AutoModel): The pre-trained SBERT model used for generating embeddings.
        window (int, optional): The number of words around the mention to consider for creating the context.
        output_dir (str, optional): Directory where the results and texts will be saved, one file per sample.
        
    Returns:
        list: A list of results from the encode_contexts function, each corresponding to a sample text.
    """
    import os

    if len(sample_texts) != len(mentions_list):
        logger.error("The lengths of sample_texts and mentions_list do not match.")
        raise ValueError("sample_texts and mentions_list must have the same length.")

    results = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    
    # Process each sample text with its corresponding mentions
    for i, (sample_text, mentions) in enumerate(zip(sample_texts, mentions_list)):
        logger.info(f"Processing sample: {sample_text[:50]}...")  # Log the beginning of the sample text for context
        encoded_results = encode_contexts(mentions, sample_text, tokenizer, model, window)
        results.append(encoded_results)
        
        if output_dir and start_index:
            # Save the encoded results
            with open(os.path.join(output_dir, f"encoded_results_{start_index + i}.txt"), 'w') as file:
                file.write(str(encoded_results))  # Writing the encoded results as a string

            # Save the corresponding sample text
            with open(os.path.join(output_dir, f"sample_text_{start_index + i}.txt"), 'w') as file:
                file.write(sample_text)  # Writing the sample text

    # Return zipped results for each sample and its encoded contexts
    return zip(sample_texts, results)

# Experimental

@audit_logger(functions_to_audit)
def get_context(text, mention, window=5):
    """
    Retrieves the context around each term mention in the text, defined in terms of a window of words,
    and keeps the original start and end positions of each mention for potential annotations.
    
    Args:
    text (str): The full text from which contexts are extracted.
    mention (tuple): A tuple containing (term, start_pos) of a mention.
    window (int): Number of words to consider around the mention for context.

    Returns:
    str: The context text around the mention.
    """
    # Define a regex pattern to split words on spaces or special characters
    pattern = re.compile(r'\s+|[-/:~*%@#^\.<>()$&\|]+')
    words = pattern.split(text)
    
    _, start_pos = mention  # Unpack the mention
    logger.info(f"Mention: {mention}: {type(mention)}") # 
    logger.info(f"Text: {text}")
    # Generate word positions
    word_positions = []
    current_pos = 0
    for word in words:
        if word:  # Skip empty strings resulting from consecutive delimiters
            word_start_pos = text.find(word, current_pos)
            word_positions.append((word, word_start_pos))
            current_pos = word_start_pos + len(word)
    logger.debug(f"Word positions: {word_positions}") # OK

    # Find the word index of the start position
    try:
        term_index = next(i for i, (_, pos) in enumerate(word_positions) if pos == start_pos)
    except Exception as e:
        logger.error(f"Problem getting term_index: {e}")
        raise
    logger.debug(f"Term index: {term_index}") # OK

    # Determine the start and end indices for the context window
    context_start_index = max(term_index - window, 0)
    context_end_index = min(term_index + window + 1, len(words))

    context = ' '.join(words[context_start_index:context_end_index])
    logger.debug(window)
    logger.debug(mention)
    logger.debug(text)
    logger.debug(context)
    return context

@audit_logger(functions_to_audit)
def get_embedding(text, tokenizer, model):
    """
    Generalizes the embedding of a string using a tokenizer and a model.

    Args:
        text (str): The text string to embed.
        tokenizer (AutoTokenizer): Tokenizer from the SBERT model.
        model (AutoModel): Pre-trained SBERT model.

    Returns:
        torch.Tensor: Embedding for the provided text.
    """
    if not isinstance(text, str):
        logger.error(f"text is not str, got {type(text)}")
        raise
    logger.debug(f'text: {text}')
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = model_output.last_hidden_state.mean(dim=1).squeeze()
    return embedding

@audit_logger(functions_to_audit)
def encode_context(mention, context, tokenizer, model):
    """
    Encodes a single text context using the provided SBERT model and tokenizer, along with position information.

    Args:
        mention (tuple): A tuple containing the term and start position in the original text.
        context (str): The text context for the term.
        tokenizer (AutoTokenizer): Tokenizer from the SBERT model.
        model (AutoModel): Pre-trained SBERT model.

    Returns:
        tuple: Tuple containing the term and the embedding.
    """
    term, _ = mention
    # Encode the context using a helper function to obtain the embedding
    embedding = get_embedding(context, tokenizer, model)
        
    return (term, embedding)

@audit_logger(functions_to_audit)
def process_and_encode_samples(sample_texts_dict, mentions_list, tokenizer, model, window=5):
    """
    Process multiple text samples with their corresponding mentions, extract contexts around mentions,
    and encode the contexts using the provided SBERT model and tokenizer. Samples are passed in batches.

    Args:
        sample_texts_dict (dict): A dictionary where keys are unique identifiers and values are text samples.
        mentions_list (list of list of dicts): Each dict contains a term, its start position, sample_text_identifier, and context.
        tokenizer (AutoTokenizer): Tokenizer from the SBERT model used for tokenizing the text.
        model (AutoModel): The pre-trained SBERT model used for generating embeddings.
        window (int, optional): The number of words around the mention to consider for creating the context.
        
    Returns:
        list: A list of results, each containing terms, their embeddings, and the sample text identifier.
    """
    results = []
    
    for mentions in mentions_list:
        for mention in mentions:
            logger.info(f"mention: {type(mention)} {mention}") # 
            sample_id = mention['sample_text_identifier']
            logger.warning(f"sample_text_identifier: {sample_id}")
            try:
                sample_text = sample_texts_dict[sample_id]
            except KeyError as e:
                logger.error(f"Error while joining mentions for {sample_id}: {e}")
                continue

            logger.warning(f"Processing sample: {sample_text[:50]}...")  # Log the beginning of the sample text for context

            encoded_results_with_id = []

            if isinstance(mention, dict):
                mention = [mention]
            else:
                for mention in mentions:
                    if not isinstance(mention, dict) or 'term' not in mention or 'start_pos' not in mention:
                        logger.error(f"Got: {mention}")
                        logger.error("Mentions must be a list of dictionaries with 'term' and 'start_pos'.")
                        continue

        for mention in mentions:
            logger.info(f"Mention: {mention}") # 
            term = mention['term']
            start_pos = mention['start_pos']
            sample_text = sample_texts_dict[sample_id]
            context = get_context(sample_text, (term, start_pos), window)
            logger.info(f"Context: {context}") # 

            # Call encode_context with the context text and mention details
            term, context_embedding = encode_context((term, start_pos), context, tokenizer, model)
            logger.info(f"term: {term}") # 
            logger.info(f"context_embedding: {type(context_embedding)}") # 
            logger.info(f"sample_id: {sample_id}") # NG

            # It's important to distinguish context embeddings from other types of embedding.
            encoded_results_with_id.append({
                'term': term,
                'start_pos': start_pos,
                'context_embedding': context_embedding,
                'sample_text_identifier': sample_id
            })

        logger.debug(f"encoded_results_with_id: {len(encoded_results_with_id)}") # OK

        results.append(encoded_results_with_id)
    logger.debug(f"results: {len(results)}") # OK

    # Return results for each sample and its encoded contexts
    return results  # sample_id gives enough information to identify the encoded results

@audit_logger(functions_to_audit)
def convert_to_serializable(obj):
    """
    Recursively convert tensors, numpy arrays, and nested tuples into serializable formats.
    
    Args:
        obj: The object to convert.
        
    Returns:
        The converted object.
    """
    if hasattr(obj, 'tolist'):
        return obj.tolist()  # Convert tensor or numpy array to list
    elif isinstance(obj, (tuple, list)):
        return [convert_to_serializable(item) for item in obj]  # Recursively convert each item in the tuple or list
    elif isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj  # Return the object as-is if no conversion is needed

@audit_logger(functions_to_audit)
def serialize_processed_data(data, output_path, data_format='parquet', overwrite='no'):
    """
    Serialize the processed data into the specified format (CSV, JSON, or Parquet).
    
    Args:
        data (list): Processed data to be serialized.
        output_path (str): Path to the output file.
        data_format (str, optional): Format to serialize data ('csv', 'json', 'parquet').
        overwrite (str, optional): File overwrite behavior ('yes', 'no', 'avoid').
        logger (Logger, optional): Logger for debugging and logging information.
        
    Returns:
        None
    """
    logger.warning(f'data: {type(data)}: len {len(data)}')
    valid_extensions = {'csv', 'json', 'parquet'}
    if '.' in output_path:
        file_extension = output_path.split('.')[-1]
        if file_extension not in valid_extensions:
            if logger:
                logger.warning(f"Invalid file extension {file_extension}. Skipping...")
            return
        data_format = file_extension
    else:
        output_path += f".{data_format}"
    
    logger.warning(f"overwrite: {overwrite}")
    if overwrite.lower() not in {'yes', 'y'}:
        if os.path.exists(output_path):
            if overwrite.lower() in {'no', 'n'}:
                if logger:
                    logger.error(f"File {output_path} already exists. Skipping...")
                return
            elif overwrite.lower() in {'avoid', 'a'}:
                base, ext = os.path.splitext(output_path)
                counter = 1
                while os.path.exists(output_path):
                    output_path = f"{base}_{counter:04d}{ext}"
                    counter += 1
                if logger:
                    logger.info(f"Output file exists. Saving to new file: {output_path}")
    
    logger.debug(f"Data to serialize: {data}")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Recursively convert all fields to serializable formats and structure rows
    structured_data = []
    for row in data:
        for mention in row:
            logger.debug(f"term: {mention['term']}")
            logger.debug(f"context_embedding: {type(mention['context_embedding'])}")
            logger.debug(f"sample_text_identifier: {mention['sample_text_identifier']}")
            mention_serializable = convert_to_serializable(mention)
            structured_data.append(mention_serializable)
            logger.debug(f"mention_serializable is now of type: {type(mention_serializable)}, len: {len(mention_serializable)}")

    if data_format == 'csv':
        with open(output_path, 'w', newline='') as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(['term', 'start_pos', 'context_embedding', 'sample_text_identifier'])
            for row in structured_data:
                csv_writer.writerow([
                    row['term'], 
                    row['start_pos'], 
                    row['context_embedding'], 
                    row['sample_text_identifier']
                ])
    elif data_format == 'json':
        with open(output_path, 'w') as output_file:
            json.dump(structured_data, output_file)
    elif data_format == 'parquet':
        df = pd.DataFrame(structured_data, columns=['term', 'start_pos', 'context_embedding', 'sample_text_identifier'])
        df.to_parquet(output_path)

@audit_logger(functions_to_audit)
def convert_sample_texts_to_dict(sample_texts):
    """
    Convert a list of raw text to a dictionary with MD5 hash keys.
    
    Args:
        sample_texts (list of str): List of raw text samples.
        
    Returns:
        dict: Dictionary with MD5 hash keys and raw text values.
    """
    sample_texts_dict = {}
    for text in sample_texts:
        md5_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        sample_texts_dict[md5_hash] = text
    return sample_texts_dict

@audit_logger(functions_to_audit)
def find_term_mentions(text, terms):
    """
    Identifies and locates terms specified in a given text, returning both the mentions of the terms and a list of these terms found uniquely.

    Args:
        text (str): The text to be scanned for the specified terms.
        terms (list of str): Terms to search within the text.

    Returns:
        tuple: Contains two elements:
            - A list of dictionaries for each term mention, each containing 'term', 'start_pos', and 'sample_text_identifier'.
            - A list of unique terms found in the text, sorted alphabetically.
    """
    if not isinstance(terms, list):
        logger.error(f"terms is type {type(terms)}, expected list.")
        raise TypeError("Expected terms to be a list")
    
    # Normalize terms and remove duplicates, prioritize longer terms first
    cleaned_terms = {term.strip().lower() for term in terms if term is not None and isinstance(term, str)}
    cleaned_terms = sorted(cleaned_terms, key=len, reverse=True)
    
    logger.debug(f"got terms: {type(terms)}") # OK
    logger.debug(f"first term: {terms[0]}")
    logger.debug(f"text: {text}") # OK
    mentions = []
    found_terms = set()

    # Match terms in the text
    for term in cleaned_terms:
        logger.debug(f"Matching '{term}' to text: {text}") # OK
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        matches = list(re.finditer(pattern, text))
        if len(matches) > 0:
            logger.debug(f"Found {len(matches)} matches") # OK
            for match in matches:
                original_text = text[match.start():match.end()]  # Extract the matched substring with original casing
                new_match = {
                    'term': original_text,
                    'start_pos': match.start()
                }
                logger.debug(f"new_match: {new_match}") # OK
                mentions.append(new_match)
                logger.debug(f"mentions after appending: {mentions}") # OK
                found_terms.add(term)  # Track found terms in lower case
                logger.debug(f"found_term: {term}") # OK

    # Sort and deduplicate mentions
    mentions.sort(key=lambda x: x['start_pos'])  # Sort by start position
    logger.debug(f"mentions (sorted): {mentions}") # OK

    # Handle overlapping mentions by retaining the longest first matched
    final_mentions = []
    last_end = -1
    for mention in mentions:
        if mention['start_pos'] > last_end:
            final_mentions.append(mention)
            last_end = mention['start_pos'] + len(mention['term'])
    
    logger.debug(f"final_mentions: {final_mentions}") # OK

    # Convert found terms set to a sorted list
    unique_terms = sorted(found_terms)
    logger.debug(f"Found {len(final_mentions)} non-overlapping mentions of unique terms.")
    return final_mentions, unique_terms

@audit_logger(functions_to_audit)
def find_mentions_in_samples(sample_texts, terms):
    """
    Process multiple text samples to find mentions of terms for each sample using Aho-Corasick algorithm.
    
    Args:
        sample_texts (list of str): A list of text samples in which to find term mentions.
        terms (list of str): A shared list of terms to search within all text samples.
        
    Returns:
        list: A list of results from the Aho-Corasick matching, each corresponding to a sample text.
    """
    results = []

    # Clean and prepare terms list
    logger.info(f"terms: {type(terms)}, len {len(terms)}")
    cleaned_terms = list(set(str(term).strip() for term in terms if term is not None))
    cleaned_terms = sorted(cleaned_terms, key=len, reverse=True)
    logger.info(f"sample_texts: {type(sample_texts)}, len {len(sample_texts)}")
    logger.info(f"cleaned_terms count: {len(cleaned_terms)}")
    logger.debug(f"cleaned_terms: {cleaned_terms}")

    # Build Aho-Corasick automaton
    automaton = ahocorasick.Automaton()
    for term in cleaned_terms:
        automaton.add_word(term, term)
    automaton.make_automaton()

    boundary_pattern = re.compile(r'\s+|[-/:~*%@#^\.<>()$&\|]+')

    for sample_text in sample_texts:
        logger.warning(f"Processing sample text: {sample_text[:50]}...")
        mentions = []
        unique_terms = set()
        for end_index, term in automaton.iter(sample_text):
            start_index = end_index - len(term) + 1
            
            # Ensure the match is surrounded by the defined boundary characters
            if (start_index == 0 or boundary_pattern.match(sample_text[start_index - 1])) and \
               (end_index == len(sample_text) - 1 or boundary_pattern.match(sample_text[end_index + 1])):
                mentions.append({
                    'term': term,
                    'start_pos': start_index,
                    'sample_text_identifier': hashlib.md5(sample_text.encode('utf-8')).hexdigest()
                })
                unique_terms.add(term)
        logger.debug(f"mentions: {mentions}")
        logger.debug(f"unique_terms: {unique_terms}")
        results.append((mentions, sorted(unique_terms)))

    return results

@audit_logger(functions_to_audit)
def normalize_candidate_definitions(candidate_definitions):
    """
    Normalize candidate definitions by removing duplicate entries and organizing them in a nested structure.
    
    Args:
    candidate_definitions (dict): The candidate definitions to normalize.
    
    Returns:
    dict: The normalized candidate definitions in a nested structure.
    """
    
    normalized_definitions = {}
    logger.debug(f"Candidate definitions to normalize:\n{format_json_topology(candidate_definitions)}")
    
    for term, contexts in candidate_definitions.items(): # term, source: {}
        logger.debug(f"term: {term}")
        logger.debug(f"contexts:\n{format_json_topology(contexts)}")
        if term not in normalized_definitions:
            normalized_definitions[term] = {}
        
        for context_hash, definitions in contexts.items():
            if not isinstance(context_hash, str):
                logger.error(f"context_hash must be a hash str, got {type(context_hash)}: {context_hash}")
                raise TypeError(f"context_hash must be a string, got {type(context_hash)}")
            logger.debug(f"context_hash: {context_hash}")
            logger.debug(f"contexts:\n{format_json_topology(definitions)}")
            
            for definition_dict in definitions:
                logger.debug(f"definition_dict:\n{format_json_topology(definition_dict)}")

                # Missing these keys is an error condition
                # May consider defaulting or skipping on error in the future
                source = definition_dict['source']
                embedding = definition_dict['embedding']
                def_text = definition_dict['definition']
                label = definition_dict['label']
                def_url = definition_dict['id']

                logger.debug(f"""
                    term: {term}
                    source: {source}
                    def_text: {def_text}
                    label: {label}
                    def_url: {def_url}""")

                if source not in normalized_definitions[term]:
                    normalized_definitions[term][source] = {}

                def_url_str = str(def_url)  # Ensure def_url is a string
                if def_url_str not in normalized_definitions[term][source]:
                    normalized_definitions[term][source][def_url_str] = {
                        "definition": def_text,
                        "embedding": embedding,
                        "sample_text_hashes": set()
                    }
                
                normalized_definitions[term][source][def_url_str]["sample_text_hashes"].add(context_hash)
                logger.debug(f'Added definition: source={source}, def_url={def_url}, def_text={def_text}, context_hash={context_hash}')
                
    # Convert sets to lists for JSON serialization
    for term in normalized_definitions:
        for source in normalized_definitions[term]:
            for def_url_str in normalized_definitions[term][source]:
                normalized_definitions[term][source][def_url_str]["sample_text_hashes"] = list(
                    normalized_definitions[term][source][def_url_str]["sample_text_hashes"]
                )
    
    logger.debug(f"Final normalized candidate definitions:\n{format_json_topology(normalized_definitions)}")
    return normalized_definitions



@audit_logger(functions_to_audit)
def collect_and_process_contexts(nodes, unique_terms, endpoints, lodc_data_path):
    contexts = []
    node_to_context_map = {}

    logger.info(f'Collect contexts from nodes')
    for node in nodes:
        if 'contexts' in node:
            for context in node['contexts']:
                context_hash = hashlib.md5(json.dumps(convert_to_serializable(context)).encode('utf-8')).hexdigest()
                contexts.append(context)
                node_to_context_map[context_hash] = node

    logger.info(f"Processing candidate definitions")
    candidate_definitions = process_candidate_definitions(contexts, unique_terms, endpoints, lodc_data_path)

    logger.info(f'Ranking embeddings and appending results to nodes')
    for context in contexts:
        context_serializable = convert_to_serializable(context)
        context_hash = hashlib.md5(json.dumps(context_serializable).encode('utf-8')).hexdigest()
        term = context['term']
        start_pos = context['start_pos']
        context_embedding = context['context_embedding']
        logger.info(f"Calling rank_embeddings")
        term_result = rank_embeddings(candidate_definitions, term, context_embedding, start_pos)
        if term_result['code'] != 'OK':
            logger.info('No usable term_result')
            continue
        
        if context_hash in node_to_context_map:
            node = node_to_context_map[context_hash]
            if 'ranked_definitions' not in node:
                node['ranked_definitions'] = []
            node['ranked_definitions'].append(term_result)
            logger.warning(f"Appended {term_result['sources']}")

    return nodes

@audit_logger(functions_to_audit)
def rank_embeddings(candidate_definition_embeddings, term, context_embedding, start_pos, min_similarity=0.1):
    if not isinstance(min_similarity, float):
        logger.error(f"min_similarity is not float, got: {type(min_similarity)}")
        raise ValueError("min_similarity must be a float")
    
    if not isinstance(term, str):
        logger.error(f"term: Expected str, got {type(term)}")
        raise ValueError("term must be a string")
    
    logger.debug(f"Got definition embeddings:\n{format_json_topology(candidate_definition_embeddings)}")

    best_per_source = {}

    try:
        # Convert context_embedding to a PyTorch tensor if it's not already
        if isinstance(context_embedding, np.ndarray):
            context_embedding = torch.tensor(context_embedding)
        context_embedding_fresh = context_embedding.unsqueeze(0)  # More efficient since this is used each loop
        logger.debug(f"Converted context embedding to PyTorch tensor: {context_embedding_fresh.shape}")
    except AttributeError as e:
        logger.error(f"Error while unsqueezing context embedding ({type(context_embedding)}): {e}")
        logger.error(f"{context_embedding}")
        raise

    for term_key, source_definitions in candidate_definition_embeddings.items():
        logger.debug(f"term_key: {term_key}")
        for source, definitions in source_definitions.items():
            logger.debug(f"Processing source: {source} with definitions:\n{format_json_topology(definitions)}")
            for def_url, def_info in definitions.items(): # def_id should be the identifier used in LODC URLs
                logger.debug(f"r_e using def_url: {def_url}")
                embedding = def_info['embedding'] # Absence is a critical error condition
                try:
                    # Convert definition_embedding to a PyTorch tensor if it's not already
                    if isinstance(embedding, np.ndarray):
                        embedding = torch.tensor(embedding)
                    definition_embedding_fresh = embedding.unsqueeze(0)
                    logger.debug(f"Converted definition embedding to PyTorch tensor: {definition_embedding_fresh.shape}")
                except AttributeError as e:
                    logger.error(f"Error while unsqueezing definition embedding ({type(embedding)}): {e}")
                    logger.error(f"Definition embedding: {embedding}")
                    continue

                try:
                    similarity = F.cosine_similarity(definition_embedding_fresh, context_embedding_fresh, dim=1).item()
                    logger.debug(f"Similarity {similarity} computed for term '{term}' from source '{source}'")
                except Exception as e:
                    logger.error(f"Error while computing similarity: {e}")
                    continue
                
                # Check if the similarity is above the threshold
                if similarity >= min_similarity:
                    logger.debug(f"Similarity is above min_similarity {min_similarity}")
                    # Update the best match for this source if it's the highest we've seen
                    if source not in best_per_source or best_per_source[source]['similarity'] < similarity:
                        best_per_source[source] = {
                            'url': def_url,
                            'similarity': similarity
                        }
                else:
                    # Log when similarity is below the threshold
                    logger.info(f"Discarded low similarity definition for term '{term}' from source '{source}'.")

    # Prepare structured result for this term
    sources_list = {source: info for source, info in best_per_source.items()}

    term_result = {
        'term': term,
        'start_pos': start_pos,
        'code': 'OK' if sources_list else 'EMPTY'
    }

    if sources_list:
        term_result['sources'] = sources_list
    
    logger.debug(f"Results for {term} at position {start_pos}: {term_result}")

    return term_result

@audit_logger(functions_to_audit)
def extract_definitions(json_data, description_keys, identifier_keys):
    """
    Extracts definitions and their corresponding identifiers from JSON data based on provided keys.

    Args:
        json_data (dict or list): The JSON data from which definitions and identifiers are to be extracted.
        description_keys (list): Paths to locate the definitions in the JSON data.
        identifier_keys (list): Paths to locate the identifiers in the JSON data.

    Returns:
        list of tuples: Pairs of extracted definitions and their corresponding identifiers.
    """
    logger.debug(f'Checking description keys: {description_keys} and identifier keys: {identifier_keys}')

    # Extract definitions and identifiers from the JSON data
    definitions = collect_values_with_key_specific_filters(json_data, description_keys)
    identifiers = collect_values_with_key_specific_filters(json_data, identifier_keys)
    logger.debug(f"definitions: {len(definitions)}")
    logger.debug(f"identifiers: {len(identifiers)}")

    # Check if both lists are populated and match in length
    if len(definitions) != len(identifiers):
        error_message = f"Mismatch or absence of data between definitions and identifiers: {definitions}, {identifiers}"
        logger.warning(error_message) # Drop edge cases and corrupted nodes like Q2516350

    # Pair each definition with its corresponding identifier in a list of tuples
    result = list(zip(definitions, identifiers))
    return result
    
@audit_logger(functions_to_audit)
def get_values_from_json_path(json_data, path):
    """
    Utilizes match_json_path to find and collect values from a JSON object by following a specified path.

    Args:
    json_data (any): The JSON data to search.
    path (list): The path to follow in the JSON structure.

    Returns:
    list: A list of values found at the specified path.
    """
    # Call match_json_path to get all matching nodes for the given path
    logger.debug(f"Matching on path: {path}")
    matches = match_json_path(json_data, path)
    
    # Extract the values from the matches
    values = [node for parent, node in matches if not isinstance(node, (dict, list))]
    
    return values

@audit_logger(functions_to_audit)
def match_json_path(json_data, path, dynamic_keys=None):
    """
    Recursively searches for and returns all nodes matching the given path in a JSON object, supporting dynamic keys.

    Args:
        json_data (dict or list): The JSON data to search.
        path (str): The path to search for in the JSON structure, with keys separated by '|'.
        dynamic_keys (set or dict): A set or dictionary of dynamic keys to recognize and handle in the path.

    Returns:
        list: A list of tuples, each tuple containing a reference to the matching node and the parent node.
    """
    if not path:
        logger.warning("path is empty, please check the calling function")
        return []

    path_list = path.split('|')
    results = []

    def recursive_search(current_data, current_path, parent):
        if not current_path:
            results.append((parent, current_data))
            return

        key = current_path[0]
        logger.debug(f"Key: {key}, current_path: {current_path}")

        if isinstance(current_data, dict):
            if key == "*" or (dynamic_keys and key in dynamic_keys):
                # Handle dynamic key matching
                for k in current_data:
                    if key == "*" or (dynamic_keys and k in dynamic_keys):
                        recursive_search(current_data[k], current_path[1:], current_data)
            elif key in current_data:
                recursive_search(current_data[key], current_path[1:], current_data)
            else:
                # Early return if the key is not found
                logger.debug(f"Key '{key}' not found in current level.")
                return
        elif isinstance(current_data, list):
            for item in current_data:
                recursive_search(item, current_path, current_data)
        else:
            # Early return if current_data is neither dict nor list
            logger.debug(f"Current data is neither dict nor list: {type(current_data)}")
            return

    logger.debug(f"Beginning recursive search on path: {path}")
    recursive_search(json_data, path_list, None)
   
    if not results: 
        logger.debug(f"No data found on path: {path}")
        logger.debug(f"Raw JSON: {str(json_data)[:80]}...")
    return results

@audit_logger(functions_to_audit)
def nest_secondary_json(primary_json, nested_data_list, key_string):
    # Convert the key string to a list of keys
    key_path = key_string.split('|')

    # Process nested data into a single dictionary
    nested_data = {}
    for nested in nested_data_list:
        logger.debug(f'Attempting to nest {nested}')
        nested_data.update(nested)
    
    # Recursive function to follow the key path and insert the nested data
    def nest_data(data, keys):
        if not keys:
            return

        key = keys[0]
        if key in data:
            if len(keys) == 1:
                logger.debug(f"Nesting at key {key}: {data}")
                # At the final key, add the resolved data
                if isinstance(data[key], dict) and 'value' in data[key] and data[key]['value'] in nested_data:
                    data[key]['resolved'] = nested_data[data[key]['value']]
            else:
                if isinstance(data[key], list):
                    for item in data[key]:
                        nest_data(item, keys[1:])
                else:
                    nest_data(data[key], keys[1:])
        else:
            logger.warning(f"Key {key} not found in data: {data}")

    # Start the nesting process from the root of the primary data
    nest_data(primary_json, key_path)
    
    return primary_json

@audit_logger(functions_to_audit)
def segregate_json_by_path(primary_json, valid_path):
    """
    Segregates the JSON elements based on whether they match the valid path.

    Args:
        primary_json (dict): The primary JSON data to segregate.
        valid_path (list): The valid path elements.

    Returns:
        tuple: (valid_elements, invalid_elements) JSON elements that match and don't match the valid path.
    """
    valid_elements = {}
    invalid_elements = {}

    for key, value in primary_json.items():
        if match_json_path({key: value}, valid_path):
            valid_elements[key] = value
        else:
            invalid_elements[key] = value

    return valid_elements, invalid_elements

@audit_logger(functions_to_audit)
def save_json(json_data, file_path, relative_path=False):
    """
    Saves JSON data to the specified file path.

    Args:
        json_data (dict): The JSON data to save.
        file_path (str): The file path where the JSON data should be saved.
    """
    if relative_path:
        file_path = local_path(file_path)
    logger.info(f"Saving to {file_path}")
    with open(file_path, 'w') as file:
        json.dump(json_data, file, indent=4)

@audit_logger(functions_to_audit)
def fetch_title_description(endpoint, uri, get_requests):
    """
    Fetches the title and description for a given URI using a specified SPARQL endpoint configuration. 
    The data is retrieved using a GET request, and the request's parameters are built dynamically based on the endpoint configuration.

    Args:
        endpoint (dict): A dictionary containing the endpoint configuration such as URL, query template, and optional parameters.
        uri (str): The URI to query.
        get_requests (function): A function to execute the GET request. It should accept a URL and headers, and return a response object.

    Returns:
        dict or None: Returns the JSON data as a dictionary if the request is successful (HTTP 200); otherwise, returns None.
    """
    # Construct query URL
    query = "DESCRIBE <{uri}>"
    url = endpoint['url']
    headers = endpoint.get('headers', {})

    # Set up parameters including any optional ones present in the endpoint configuration
    params = {
        'query': query,
        **{k: endpoint[k] for k in [
            'default-graph-uri', 'format', 'timeout', 'signal-void', 'signal_unconnected'
        ] if k in endpoint}
    }

    # URL encode the parameters
    encoded_params = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    full_url = f"{url}?{encoded_params}"
    logger.critical(f"full_url: {full_url}")
    # Perform the GET request using the provided get_requests function
    try:
        response = get_requests(full_url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"HTTP {response.status_code} received for URL: {full_url}")
            return None
    except RuntimeError as e:
        print(f"Runtime error during fetching data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during fetching data: {e}")
        return None

@audit_logger(functions_to_audit)
def load_json(file_path, relative_path=False):
    if relative_path:
        file_path = local_path(file_path)
    logger.info(f"Loading from {file_path}")
    with open(file_path, 'r') as file:
        return json.loads(file.read())

@audit_logger(functions_to_audit)
def save_json_response(endpoint, term, json_data, data_path):
    """
    Prepares the directory, constructs the file path using the term, and saves the JSON data.

    Args:
        endpoint (dict): Information about the endpoint to process.
        term (str): The term to use in the file name.
        json_data (dict): The JSON data to save.
        data_path (str): Base directory path for saving JSON files.
    """
    endpoint_dir = os.path.join(data_path, endpoint['name'])
    if not os.path.exists(endpoint_dir):
        os.makedirs(endpoint_dir)
    save_json_with_term(json_data, endpoint_dir, term)

@audit_logger(functions_to_audit)
def save_json_with_term(json_data, dir, term):
    """
    Constructs the file path using the term and saves the JSON data.

    Args:
        json_data (dict): The JSON data to save.
        dir (str): The directory path where the JSON data should be saved.
        term (str): The term to use in the file name.
    """
    term_lower_fname = term.lower().replace(' ', '_')
    file_path = os.path.join(dir, f"{term_lower_fname}.json")
    save_json(json_data, file_path)

@audit_logger(functions_to_audit)
def set_in_path(data, path, value):
    logger.info(f'set_in_path: data:{type(data)} {len(data)}, {path}, value: {type(value)} {len(value)}')
    for key in path[:-1]:
        data = data.setdefault(key, {})
    data[path[-1]] = value

@audit_logger(functions_to_audit)
def fetch_urls(primary_json, endpoint):
    urls = get_values_from_json_path(primary_json, endpoint['locator_path'])
    if not urls[0]:
        logger.debug(f"No secondary URLs found to fetch at {endpoint['locator_path']}")
    return urls

@audit_logger(functions_to_audit)
def fetch_json_data(url, json_path, visited_urls=None, get_requests=0):
    """
    Fetches JSON data from the specified URL.

    Args:
        url (str): The URL to fetch the JSON data from.
        json_path (str): The path to append to the URL for fetching the JSON data.
        visited_urls (set, optional): A set of already visited URLs to avoid repeated processing. Defaults to None.
        get_requests (int, optional): Current count of GET requests for managing API calls. Defaults to 0.

    Returns:
        dict: The fetched JSON data, or None if the fetch fails.
    """
    if visited_urls is None:
        visited_urls = set()

    identifier = url.split('/')[-1]
    final_url = f"{json_path}/{identifier}.json"
    logger.debug(f"Attempting secondary JSON fetch from {final_url}")

    if final_url not in visited_urls:
        response, get_requests = get_request(final_url, get_requests)
        visited_urls.add(final_url)
        if response.status_code == 200:
            logger.debug(f'Successful fetch from {url}')
            return response.json()
        else:
            logger.warning(f'Failed to fetch from {url} with status {response.status_code}')
            return None
    else:
        logger.debug('URL already searched or ignored, skipping')
    
    return None


# Keyphrase_extraction

@audit_logger(functions_to_audit)
def refine_phrase(phrase, ignore_set=None, stop_words=None):
    """
    Refine phrases by removing stop words, ambiguous terms, attaching compounds to their roots, 
    and stemming or making terms singular. Optionally ignores phrases already encountered.
    
    Parameters:
    - phrase (str): The phrase to refine.
    - ignore_set (set): Optional set of phrases to ignore.
    - stop_words (set): Optional set of stop words to remove.
    
    Returns:
    - tuple: A list of refined phrases and a set of new unique raw phrases encountered.
    """
    if ignore_set is None:
        ignore_set = set()
    
    if stop_words is None:
        stop_words = nlp.Defaults.stop_words  # Use SpaCy's default stop words if none provided

    # Remove newlines and decode URL-encoded characters in the phrase
    cleaned_phrase = unquote(phrase.replace('\n', ' '))

    if not cleaned_phrase.strip() or cleaned_phrase in ignore_set:
        return [], []

    # Tokenize the phrase using SpaCy
    doc = nlp(cleaned_phrase)

    # Remove stop words and other specified words
    tokens = [token for token in doc if token.text.lower() not in stop_words]

    # Remove short and ambiguous terms
    filtered_tokens = [token for token in tokens if len(token.text) > 2]

    # Collect raw phrases for return
    new_raw_phrases = [' '.join(token.text for token in filtered_tokens[i:j+1]) 
                       for i in range(len(filtered_tokens)) 
                       for j in range(i, len(filtered_tokens))]
    
    new_raw_phrases = set(new_raw_phrases) - ignore_set  # Remove ignored phrases and duplicates

    # Stem and singularize terms
    singular_phrases = [inflector.singular_noun(phrase) if inflector.singular_noun(phrase) else phrase for phrase in new_raw_phrases]

    # Remove duplicates and return
    return list(set(singular_phrases)), new_raw_phrases

@audit_logger(functions_to_audit)
def extract_keyphrases(text, ignore_set=None):
    """
    Extract and refine keyphrases from given text, updating ignore set in place.
    Includes detailed debugging information using a logger.
    
    Parameters:
    - text (str): The text from which to extract keyphrases.
    - ignore_set (set): Optional set of phrases to ignore.

    Returns:
    - tuple: A list of unique refined keyphrases and a list of new unique raw phrases.
    """
    logger.debug(f"Processing text: {text[:50]}...")  # Log the beginning of the text to avoid long logs
    if ignore_set is None:
        ignore_set = set()  # Initialize ignore_set if not provided

    if not text:  # Handle empty text fields
        logger.info("Empty text received.")
        return [], []

    # Decode URL-encoded characters in the text
    text = unquote(text)
    try:
        doc = nlp(text)
    except Exception as e:
        logger.error(f"Failed to process text with nlp: {str(e)}")
        return [], []  # Return empty lists if NLP processing fails

    keyphrases = []
    new_raw_phrases = []

    for chunk in doc.noun_chunks:
        try:
            refined, new_raw = refine_phrase(chunk.text, ignore_set)
        except Exception as e:
            logger.warning(f"Exception while refining '{chunk.text}':\n{e}\nSkipping...")
            continue
        if refined:  # Ensure refined phrases are not empty
            keyphrases.extend(refined)
        if new_raw:  # Update ignore_set with new raw phrases encountered
            ignore_set = ignore_set.union(new_raw)
            logger.debug(f"Now ignoring '{new_raw}'")

    # Remove duplicates from the keyphrases before returning
    keyphrases = list(set(keyphrases))
    ignore_set = list(set(ignore_set))

    logger.debug(f"Final keyphrases: {keyphrases}")
    logger.debug(f"Updated ignore set: {ignore_set}")

    return keyphrases, ignore_set

@audit_logger(functions_to_audit)
def get_last_processed_chunk(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r') as file:
            last_processed = file.read().strip()
            return int(last_processed) if last_processed.isdigit() else 0
    return 0

@audit_logger(functions_to_audit)
def set_last_processed_chunk(state_file, chunk_index):
    with open(state_file, 'w') as file:
        file.write(str(chunk_index))

@audit_logger(functions_to_audit)
def set_and_ensure_path(base_path, path_extension):
    full_path = os.path.join(base_path, path_extension)
    os.makedirs(full_path, exist_ok=True)
    return full_path

@audit_logger(functions_to_audit)
def create_dataframe_from_values(dataframes_dict, save_to_csv=False, output_filename="output.csv", include_filename=False):
    # Set to collect unique values and their filenames from all DataFrames
    unique_values = set()
    values_with_filenames = []

    # Collect values from each DataFrame
    for filename, df in dataframes_dict.items():
        for value in df.values.flatten():
            unique_values.add(value)
            if include_filename:
                values_with_filenames.append((value, filename))

    # Create a new DataFrame from the set of unique values
    unique_values_df = pd.DataFrame(list(unique_values), columns=['Unique Values'])

    if include_filename:
        # Create a DataFrame with values and filenames
        values_with_filenames_df = pd.DataFrame(values_with_filenames, columns=['Unique Values', 'Filename'])
        # Split the filename on the first underscore and discard the filename
        split_filenames = values_with_filenames_df['Filename'].str.split('_', n=1, expand=True)
        values_with_filenames_df['label'] = split_filenames[0]
        values_with_filenames_df['field'] = split_filenames[1]
        # Drop the 'Filename' column
        values_with_filenames_df.drop(columns=['Filename'], inplace=True)
        unique_values_df = values_with_filenames_df[['label', 'field', 'Unique Values']].drop_duplicates()

    # Check whether to save to CSV or return the DataFrame
    if save_to_csv:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        unique_values_df.to_csv(output_filename, index=False, header=False)
        logger.info(f"Data saved to {output_filename}")

    return unique_values_df

@audit_logger(functions_to_audit)
def process_csv_in_chunks(input_filename, output_dir, chunk_size=5, max_chunks=None):
    state_file = os.path.join(output_dir, "process_csv_in_chunks_state.txt")
    logger.debug(f"Working from directory: {output_dir}")

    # Retrieves the index of the last processed chunk from state file
    last_processed_chunk = get_last_processed_chunk(state_file) 
    all_keyphrases = set()
    total_rows = sum(1 for _ in open(input_filename, 'r'))  # Count the total rows in the file
    # Calculate max chunks dynamically
    actual_max_chunks = min(max_chunks if max_chunks else float('inf'), -(-total_rows // chunk_size))  
    
    # Initialize tqdm with the correct range starting from the next chunk after the last processed
    with tqdm(total=actual_max_chunks, desc="Processing CSV Chunks") as pbar:
        for chunk_index, chunk in enumerate(
            pd.read_csv(input_filename, chunksize=chunk_size, header=None), 
            start=last_processed_chunk):
            if chunk_index >= last_processed_chunk + actual_max_chunks:
                logger.info("Max chunks reached")
                break

            logger.info(f"Processing chunk {chunk_index}")
            local_keyphrases = set()
            for index, row in chunk.iterrows():
                text = row[0]  # Assuming text is in the first column
                keyphrases, _ = extract_keyphrases(text)
                logger.debug(f'Found keyphrases: {keyphrases}')
                local_keyphrases.update(keyphrases)

            # Save current chunk's keyphrases to a file
            chunk_output_dir = os.path.join(output_dir, "chunks")
            os.makedirs(chunk_output_dir, exist_ok=True)
            with open(os.path.join(chunk_output_dir, f"chunk_{chunk_index}.csv"), "w") as file:
                for phrase in local_keyphrases:
                    file.write(f"{phrase}\n")

            all_keyphrases.update(local_keyphrases)
            set_last_processed_chunk(state_file, chunk_index)  # Update the state file with the last processed chunk index

            # Update the tqdm progress bar
            pbar.update(1)

    # Save all keyphrases to a file without header and one phrase per line
    with open(f"{output_dir}_all.csv", "w") as file:
        for phrase in all_keyphrases:
            file.write(f"{phrase}\n")

@audit_logger(functions_to_audit)
def load_csv_files(directory_path):
    logger.info(f"Checking for text in:\n'{directory_path}'")
    # Prepare a dictionary to hold filenames and DataFrames
    dataframes_dict = {}
    
    # Generate a list of all csv files in the directory and subdirectories
    csv_files = [os.path.join(root, file)
                 for root, dirs, files in os.walk(directory_path)
                 for file in files if file.endswith('.csv')]
    logger.info(f"Found {len(csv_files)} csv files to process")
    
    # Iterate over the list of file paths
    for file_path in csv_files:
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        # Try to read the CSV file while handling problematic quotes
        try:
            df = pd.read_csv(file_path, delimiter='\t', header=None, quotechar='"')
        
        except Exception as e:
            logger.error(f"Error while parsing {file_path}: {e}")
            continue
        
        # Drop rows with any NaN values
        df_clean = df.dropna(how='any')
        # Store the cleaned DataFrame with the filename as the key
        dataframes_dict[filename] = df_clean
    
    return dataframes_dict




@audit_logger(functions_to_audit)
def get_neo4j_connection(uri=None, user=None, password=None, use_env=False, max_pool_size=50, connection_timeout=30):
    """
    Establish and return a connection to a Neo4j database with connection pooling.
    
    Args:
    uri (str): The URI of the Neo4j database, e.g., 'bolt://localhost:7687'
    user (str): The username for the Neo4j database
    password (str): The password for the Neo4j database
    use_env (bool): Whether to use environment variables for credentials
    max_pool_size (int): The maximum number of connections in the pool
    connection_timeout (int): The connection timeout in seconds
    
    Returns:
    neo4j.GraphDatabase.driver: The Neo4j driver instance with connection pooling
    """
    if use_env:
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')

        if not uri or not user or not password:
            raise ValueError("Environment variables for Neo4j credentials are not set properly.")
    elif not uri or not user or not password:
        raise ValueError("URI, user, and password must be provided if use_env is False.")

    driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=max_pool_size, connection_acquisition_timeout=connection_timeout)

    return driver


        
@audit_logger(functions_to_audit)
def run_transaction(driver, transaction_function, database, parameters=None, return_neo4j_result_object=False):           
    """
    Execute a given transaction function using the provided Neo4j driver within a transaction.
    
    Args:
    driver (neo4j.GraphDatabase.driver): The Neo4j driver instance.
    transaction_function (function): The function to be executed within the transaction.
    database (str): The database to run the transaction against.
    parameters (dict, optional): A dictionary of parameters for the transaction. Defaults to None.
    return_neo4j_result_object (bool, optional): If True, returns the neo4j.Result object. Defaults to False.
    
    Returns:
    list or neo4j.Result: The result of the transaction function.
    """
    logger.debug(f"database: {database}, parameters: {parameters}")
    if parameters is None:
        parameters = {}
    with driver.session(database=database if database else None) as session:
        result = session.read_transaction(transaction_function, parameters)
        if return_neo4j_result_object:
            return result
        else:
            return result
        
# Neo4j
   
@audit_logger(functions_to_audit)
def format_value(value, indent=0):
    if isinstance(value, dict):
        return format_dict(value, indent)
    elif isinstance(value, list):
        if len(value) > 10:  # Arbitrary threshold for long lists
            return f"type={type(value)}, length={len(value)}"
        else:
            return format_list(value, indent)
    elif isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, torch.Tensor):
        return f"type={type(value)}, size={value.size()}"
    elif isinstance(value, np.ndarray):
        return f"type={type(value)}, shape={value.shape}"
    else:
        return repr(value)

@audit_logger(functions_to_audit)
def format_dict(d, indent=0):
    items = []
    for key, value in d.items():
        items.append(f"{' ' * indent}{key}: {format_value(value, indent + 2)}")
    return "{\n" + ",\n".join(items) + "\n" + " " * (indent - 2) + "}"

@audit_logger(functions_to_audit)
def format_list(lst, indent=0):
    items = [f"{' ' * indent}{format_value(value, indent + 2)}" for value in lst]
    return "[\n" + ",\n".join(items) + "\n" + " " * (indent - 2) + "]"

@audit_logger(functions_to_audit)
def format_mentions(mentions):
    formatted_mentions = '\n'.join(
        f"  {mention_key}: {format_value(mention_value, 2)}"
        for mention in mentions
        for mention_key, mention_value in mention.items()
    )
    return formatted_mentions

@audit_logger(functions_to_audit)
def format_node(node):
    return format_dict(node)

@audit_logger(functions_to_audit)
def format_top_mentions(top_mentions):
    logger.critical('Function audit: this function is still used.')
    raise
    formatted_mentions = []
    for mention, similarity, node_id, id_field in top_mentions:
        formatted_mention = '\n'.join(
            f"  {mention_key}: {format_value(mention_value, 2)}"
            for mention_key, mention_value in mention.items()
        )
        formatted_mentions.append(f"{id_field}: {node_id} {{\n{formatted_mention}\n}}")
    return '\n\n'.join(formatted_mentions)

@audit_logger(functions_to_audit)
def get_node_id(node, label, node_labels_and_keys):
    if label in node_labels_and_keys:
        identifier_field = node_labels_and_keys[label]['identifier']
        if 'n' in node:
            node = node['n']
            logger.info(f"Unpacked node = node['n']")
        if identifier_field in node:
            return node[identifier_field], identifier_field
        else:
            logger.warning(f"Node missing identifier '{identifier_field}' for label '{label}': {node.keys()}")
            return None, None
    else:
        logger.warning(f"Unknown label: {label}")
        return None, None

@audit_logger(functions_to_audit)
def has_non_empty_mentions(node):
    if 'n' not in node:
        return False
    for key, value in node['n'].items():
        if key.endswith('_mentions') and value:
            return True
    return False

@audit_logger(functions_to_audit)
def choose_random_nodes(label_key, property_name, driver, database, result_count):
    # Get a random node with a specific key and any value in a specific field
    # Should ignore previously processed fields based on the presence of one or more LODC_DEFINITION relations
    query = f'''
    MATCH (n:{label_key})
    WHERE n.{property_name} IS NOT NULL
    AND NOT EXISTS {{
        MATCH (n)-[r:LODC_DEFINITION]->()
        WHERE r.field = '{property_name}'
    }}
    WITH n, rand() AS random
    ORDER BY random
    LIMIT {result_count}
    RETURN n
    '''
    logger.warning(f"{query}")

    # Run the query
    result = run_query(driver, query, database)
    return result

@audit_logger(functions_to_audit)
def generate_merge_relationship_query(identifier, uri, relationship_properties, id_field, window, source):
    """
    Generates a Cypher query to merge nodes with a relationship that includes various properties.
    
    Args:
    identifier (str): The identifier to match the first node.
    uri (str): The substring to match the URI of the second node.
    relationship_properties (dict): Properties to include in the relationship.
    id_field (str): The field name of the identifier.
    window (int): The window size to include in the relationship.
    
    Returns:
    str: The generated Cypher query.
    """
    lodc_labels = {
        'Wikidata': '`http://wikiba.se/ontology#Item`',
        'DBPedia': 'Resource'
        # FIXME Getty not appearing
        # FIXME These should be endpoint definition items
    }

    properties_str = ', '.join([f"{key}: {repr(value)}" for key, value in relationship_properties.items()])
    
    merge_query =  f"""
    MATCH (n1) 
    WHERE n1.{id_field} = '{identifier}'
    WITH n1
    MATCH (n2:{lodc_labels[source]}) 
    WHERE n2.uri = '{uri}'
    WITH n1, n2
    MERGE (n1)-[:LODC_DEFINITION {{
        {properties_str},
        window: {window},
        timestamp: timestamp()
    }}]->(n2)
    RETURN *;
    """

    logger.info(f"merge_query:\n{merge_query}")
    
    return merge_query

@audit_logger(functions_to_audit)
def format_json_topology(json_obj):
    if not isinstance(json_obj, (dict, list)):
        logger.error(f"json_obj: Expected dict or list, got {type(json_obj)}")
        raise ValueError("json_obj must be a dict or list")
    topology = get_json_topology(json_obj)
    parsed = json.loads(str(topology).replace("'", "\""))
    json_repr = json.dumps(parsed, indent=4)
    return json_repr

def create_query_from_mention(
        mention, node, label, field, endpoints, window, node_labels_and_keys, similarity_threshold):
    """
    Process a single mention by generating and executing RDF import and merge relationship queries.

    Args:
        mention (dict): The mention dictionary containing necessary fields.
        field (str): The field name to look for mentions within each node.
        endpoints (list): List of endpoints containing data paths and source names.
        window (int): The window size for generating the merge relationship query.

    Returns:
        None
    """
    if not 'LODC_definition_similarity' in mention: 
        topo = format_json_topology(mention)
        ref_topo = """{
    "term": "str",
    "start_pos": "int",
    "context": "str",
    "context_embedding": "Tensor"
}"""
        if topo != ref_topo:    
            logger.critical(f"Unrecognized invalid mention topology:\n{topo}") # OK
        return None, None
    term = mention['term']
    start_pos = mention['start_pos']
    
    cosine_similarity = mention['LODC_definition_similarity']
    source = mention['LODC_definition_source']
    lodc_entity_id = mention['LODC_definition'] # This should be the identifier used in LODC URLs
    
    if cosine_similarity < similarity_threshold:
        logger.error(f"Low-similarity definition not previously filtered")
        raise

    node_identifier, id_field = get_node_id(node, label, node_labels_and_keys)

    if node_identifier is None:
        logger.warning(f"Attempted to create a query based on an unknown field.")
        return

    # Get the base URL from the endpoint based on the source
    base_url = next((endpoint['data_path'] for endpoint in endpoints if endpoint['name'] == source), None)
    logger.debug(f"base_url: {base_url}") # OK
    if not base_url:
        logger.warning(f"No endpoint found for source: {source}")
        return

    # Generate and execute the RDF import query
    rdf_query = generate_import_rdf_query(base_url, lodc_entity_id, 'ttl', 'en')

    # Generate and execute the merge relationship query
    relationship_properties = {
        'term': term,
        'field': field,
        'start_pos': start_pos,
        'cosine_similarity': cosine_similarity
    }

    merge_query = generate_merge_relationship_query(node_identifier, lodc_entity_id, relationship_properties, id_field, window, source)
    return rdf_query, merge_query

@audit_logger(functions_to_audit)
def get_mentions_from_node(node, field, require_similarity=False):
    """
    Retrieve mentions from a node based on the specified field and an optional flag
    to filter only mentions with a similarity score.

    Args:
        node (dict): The node dictionary containing mentions.
        field (str): The field name to look for mentions within the node.
        require_similarity (bool): If True, only include mentions with a similarity score.

    Returns:
        list: A list of mentions matching the criteria.
    """
    logger.debug(f"node: {node}") # OK
    mentions_key = f'{field}_mentions'
    mentions = node.get(mentions_key, [])

    if require_similarity:
        mentions = [mention for mention in mentions if 'LODC_definition_similarity' in mention]
    if mentions:
        logger.warning(f"mentions: {len(mentions)}")
        logger.warning(f"{format_mentions(mentions)}")
    return mentions

@audit_logger(functions_to_audit)
def remove_nodes_without_mentions(nodes):
    deleted_nodes = 0
    for node in nodes[:]:  # Use a copy of the list for safe iteration
        if not has_non_empty_mentions(node):
            logger.debug(f"Deleting node with no mentions: {node}")
            nodes.remove(node)
            deleted_nodes += 1
    logger.warning(f"Deleted {deleted_nodes} nodes with no mentions.")

@audit_logger(functions_to_audit)
def generate_import_rdf_query(base_data_url, entity_url, format, language_filter=None):
    """
    Generates a Cypher query to import RDF data from a given URL.
    
    Args:
    base_url (str): The base URL to fetch RDF data from.
    entity_id (str): The entity ID to fetch.
    format_key (str): The key for the RDF format (e.g., 'ttl').
    language_filter (str): Optional language filter.
    
    Returns:
    str: The generated Cypher query.
    """

    entity_id = entity_url.split('/')[-1] 
    if not base_data_url.endswith('/'): base_data_url += '/'
    url = f"{base_data_url}{entity_id}.{format}"
    logger.debug(f"url: {url}") # OK

    format_names = {
        'ttl': 'Turtle'
    }

    if language_filter:
        rdf_query = f"CALL n10s.rdf.import.fetch('{url}', '{format_names[format]}', {{languageFilter: '{language_filter}'}});"
    else:
        rdf_query = f"CALL n10s.rdf.import.fetch('{url}', '{format_names[format]}');"

    logger.info(f"rdf_query:\n{rdf_query}")

    return rdf_query

@audit_logger(functions_to_audit)
def unpack_node(node, max_iterations=10):
    """
    Unpack the node if it contains the key 'n' or 'node_raw' and log the action.
    Raise an error if there are other labels in the node which would be overwritten.
    Cap the number of iterations to avoid infinite loops and raise an error if the maximum is reached.

    Args:
        node (dict): The node to potentially unpack.
        max_iterations (int): Maximum number of iterations to perform unpacking.

    Returns:
        dict: The fully unpacked node.

    Raises:
        ValueError: If unpacking would overwrite existing keys or if the maximum number of iterations is reached.
    """
    iterations = 0

    while iterations < max_iterations:
        if 'n' in node:
            if any(key in node for key in node['n'] if key != 'n'):
                raise ValueError("Unpacking 'n' would overwrite existing keys in the node")
            node = node['n']  # Unpack node
            logger.debug(f"Unpacked node['n'] to node")
        elif 'node_raw' in node:
            if any(key in node for key in node['node_raw'] if key != 'node_raw'):
                raise ValueError("Unpacking 'node_raw' would overwrite existing keys in the node")
            node = node['node_raw']  # Unpack nested node_raw
            logger.debug(f"Found and corrected nested 'node_raw' field in node")
        else:
            break  # Exit loop if neither 'n' nor 'node_raw' are present
        iterations += 1

    if iterations >= max_iterations:
        raise RuntimeError(f"Maximum iterations ({max_iterations}) reached during unpacking")

    return node

# Network

@audit_logger(functions_to_audit)
def get_request(url, get_requests=0, headers=None):
    try:
        response = requests.get(url, headers=headers)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get data from {url}: {e}")
        raise RuntimeError(f"Failed to get data due to an error: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}") from e
    get_requests +=1
    return response, get_requests # Second response redundant, remove TODO

class URLTracker:
    def __init__(self):
        self.visited_urls = set()
    
    def add_url(self, url):
        """Add a single URL to the set of visited URLs."""
        self.visited_urls.add(url)

    def add_urls(self, urls):
        """Add multiple URLs from a set or the keys from a dictionary to the set of visited URLs.

        Args:
            urls (set or dict): A set of URLs or a dictionary where keys are URLs to be added.
        """
        if isinstance(urls, set):
            self.visited_urls.update(urls)
        elif isinstance(urls, dict):
            self.visited_urls.update(urls.keys())
        else:
            raise TypeError("Expected urls to be a set or a dict, got {}".format(type(urls).__name__))
    
    def remove_url(self, url):
        """Remove a URL from the set of visited URLs."""
        self.visited_urls.discard(url)
    
    def __contains__(self, url):
        """Check if a URL is in the set of visited URLs."""
        return url in self.visited_urls
    
    def __iter__(self):
        """Return an iterator over the set of visited URLs."""
        return iter(self.visited_urls)
    
    def clear(self):
        """Clear all URLs from the set of visited URLs."""
        self.visited_urls.clear()

# RDF Handling

@audit_logger(functions_to_audit)
def handle_ontology_url(url, get_requests, ontology_path, g=None):
    """
    Processes an ontology URL using RDF and SPARQL to extract data.
    
    Args:
        url (str): The ontology URL to query.
        logger: Logging object to output debug information.
        get_requests (int): Counter for the number of GET requests made.
        ontology_path (str): Path to the ontology file.
        g (rdflib.Graph, optional): RDF graph object, if not provided a new one will be created.

    Returns:
        Tuple[Dict, int]: A dictionary with extracted data and the updated GET request count.
    """
    if not g:
        logger.debug(f"RDF Graph not supplied, attempting to generate")
        g = rdflib.Graph()
        try:
            g.parse(ontology_path)
        except Exception as e:
            logger.error(f"Failed to parse ontology from {ontology_path}: {e}")
            return {}, get_requests
    
    try:
        qres = g.query(f"SELECT ?property ?hasValue WHERE {{ <{url}> ?property ?hasValue }}")
        data = {str(row.property): str(row.hasValue) for row in qres}
        get_requests += len(qres)
        logger.debug(f'Ontology data extracted for {url}' if data else f'No data found in ontology for {url}')
    except Exception as e:
        logger.error(f"Error querying ontology for URL {url}: {e}")
        data = {}

    return data, get_requests

@audit_logger(functions_to_audit)
def add_best_lodc_definitions(node, node_label, node_labels_and_keys, endpoints, lodc_data_path, 
                              tokenizer, model, terms, similarity_threshold=0.2, window=50):
    # TODO ensure that low-similarity definitions are filtered as far upstream as possible.
    mentions_added = 0
    if node_label in node_labels_and_keys:
        logger.debug(f"Found node_label: '{node_label}'")
        fields = node_labels_and_keys[node_label]['fields']
        if not fields:
            logger.warning(f"No fields found for node_label {node_label}")
            return

        for field in fields:
            logger.debug(f"Trying to add LODC definition to '{field}'")

            mentions_key = f'{field}_mentions'
            text = node[field]

            mentions = find_mentions_and_context(text, terms, endpoints, lodc_data_path, window)

            if mentions:
                logger.debug(f"found {len(mentions)} mentions in field: '{field}'")
                mentions_filtered = filter_mentions_by_longest_term(mentions, endpoints, lodc_data_path, tokenizer, model, similarity_threshold)
                logger.debug(f"mentions_key {mentions_key} filtered: {len(mentions_filtered)}")
            else:           
                logger.info(f"Found no mentions in field: '{field}'")
                continue # No further processing needed for this field
            
            node[mentions_key] = mentions_filtered

            for mention in mentions_filtered:
                term = mention['term']
                start_pos = mention['start_pos']
                context_embedding = mention['context_embedding']

                logger.debug(f"Processing candidate definitions for '{term}' at position {start_pos}")
                normalized_candidate_definitions = process_candidate_definitions(mention, endpoints, lodc_data_path, tokenizer, model)
                logger.debug(f"Got normalized_candidate_definitions.\nDoes it contain a proper LODC URL identifier?\n{format_json_topology(normalized_candidate_definitions)}")

                if term in normalized_candidate_definitions:
                    for source, definitions in normalized_candidate_definitions[term].items():
                        for def_url, def_info in definitions.items():
                            logger.debug(f"def_info:\n{format_json_topology(def_info)}")
                            logger.debug(f"Definition URL: {def_url}, Source: {source},\nDefinition: \"{def_info['definition']}\"")
                            if 'embedding' not in def_info:
                                logger.error(f"Missing embedding for Definition ID: {def_url}, Source: {source}")
                                continue
                else:
                    logger.debug(f"term '{term}' not in normalized_candidate_definitions")

                logger.debug(f"Calling rank_embeddings with\n{format_json_topology(normalized_candidate_definitions)}")
                term_result = rank_embeddings(normalized_candidate_definitions, term, context_embedding, start_pos, min_similarity=similarity_threshold)

                # Update mention with LODC definition and source
                if 'sources' in term_result and term_result['sources']:
                    logger.debug(f"term_result: {term_result}")
                    best_source_key, best_source = max(term_result['sources'].items(), key=lambda item: item[1]['similarity'])

                    best_similarity = best_source['similarity']
                    if best_similarity < similarity_threshold:
                        logger.warning(f"Low-similarity definition not previously filtered")
                        continue
                    mention['LODC_definition_similarity'] = best_similarity
                    mention['LODC_definition'] = best_source['url'] # This should be the identifier used in LODC URLs
                    logger.debug(f"Using mention['LODC_definition']: {mention['LODC_definition']}")
                    mention['LODC_definition_source'] = best_source_key
                    
                else:
                    logger.debug(f"'sources' not in term_result or not term_result['sources']")

            # Remove mentions without definitions
            node[mentions_key] = [mention for mention in node[mentions_key] if 'LODC_definition' in mention]
            mentions_added += len(node[mentions_key])
        else:
            logger.debug(f"node_label '{node_label}' not found")
    
    return mentions_added

@audit_logger(functions_to_audit)
def deduplicate_terms(terms):
    normalized_terms = set(term.lower() for term, _, _ in terms)
    return list(normalized_terms)

@audit_logger(functions_to_audit)
def extract_terms_and_expansions(terms_df, columns):
    """
    Extracts unique terms and their expansions from specified columns in a DataFrame.

    Args:
        terms_df (DataFrame): The pandas DataFrame from which to extract terms.
        columns (list of str): Column names in the DataFrame where terms are located.

    Returns:
        list: A list of unique terms extracted from the specified columns.
    """
    logger.debug(f"Extracting terms from the dataframe from the following columns: {columns}")
    if not set(columns).issubset(terms_df.columns):
        missing_cols = set(columns) - set(terms_df.columns)
        raise ValueError(f"DataFrame does not contain the following columns: {', '.join(missing_cols)}")
    unique_terms = set()
    for column in columns:
        unique_terms.update(terms_df[column].drop_duplicates().tolist())
    logger.debug(f"Unique terms extracted: {unique_terms}")
    return list(unique_terms)

@audit_logger(functions_to_audit)
def get_term_positions(text, words):
    """ Helper function to get the starting position of each term in the text. """
    positions = []
    start = 0
    for word in words:
        start = text.find(word, start)
        positions.append(start)
        start += len(word)
    return positions

@audit_logger(functions_to_audit)
def find_term_index(positions, char_position):
    """ Find the index of the word that includes the character position. """
    for index, pos in enumerate(positions):
        if pos > char_position:
            return max(index - 1, 0)
    return len(positions) - 1

@audit_logger(functions_to_audit)
def load_and_extract_definitions(term, endpoint, base_path, file=None):
    """
    Loads JSON data for a specified term from a constructed file path and extracts definitions based on a given configuration.

    Args:
        term (str): The term for which to load and extract definitions.
        endpoint (dict): Contains endpoint configurations including the name for constructing the file path.
        base_path (str): The base directory path where JSON files are stored.

    Returns:
        list of dicts or None: Returns a list of dictionaries each containing a definition and its source and identifier information,
                               or None if no definitions are extracted or if an error occurs.
    """
    if file:
        file_path = file
        logger.debug(f'Using supplied file: {file_path}')
    else:
        # Construct the file path
        term_lower_fpath = term.lower().replace(' ', '_')
        intermediate_str = f"{endpoint['name']}\\{term_lower_fpath}.json"
        file_path = os.path.join(base_path, intermediate_str)
        logger.debug(f'Constructed file path: {file_path}')

    # Check if the file exists
    if not os.path.exists(file_path):
        logger.debug(f'No JSON file found for {term} at {file_path}. Skipping...')
        return None

    if os.path.getsize(file_path) <= 4:
        logger.debug("JSON file appears to be empty, skipping.")
        return None

    json_data = load_json(file_path)

    # Extract definitions using the configured paths
    logger.debug(f"Attempting to extract definitions for {term}")
    logger.debug(f"endpoint['description_keys']: {endpoint['description_keys']}")
    description_keys = endpoint['description_keys']
    logger.debug(f"description_keys: {description_keys}")
    identifier_keys = endpoint['identifier_keys']
    logger.debug(f'identifier_keys: {identifier_keys}')
    label_keys = endpoint['label_keys']
    logger.debug(f'label_keys: {label_keys}')
    
    # Use search_json_for_keys instead of extract_definitions
    results_keys = endpoint['results_keys']
    logger.debug(f"results_keys: {results_keys}")
    definitions = []
    for res_key in results_keys:
        logger.debug(f"res_key: {res_key}")
        results = search_json_for_keys(json_data, res_key)
    
        for result in results:
            for desc_key, id_key, label_key in zip(description_keys, identifier_keys, label_keys):
                logger.debug(f"Searching with desc_key: {desc_key} and id_key: {id_key}")
                descriptions = search_json_for_keys(result, desc_key)
                identifiers = search_json_for_keys(result, id_key)
                labels = search_json_for_keys(result, label_key)
                if not descriptions or not identifiers or len(descriptions) != len(identifiers):
                    logger.debug(f"No valid results for this combination")
                    continue
                logger.debug(f"descriptions: {descriptions}")
                logger.debug(f"identifiers: {identifiers}") # Must be LODC URL compatible
                logger.debug(f"labels: {labels}")
                
                for description, identifier, label in zip(descriptions, identifiers, labels):
                    definitions.append({
                        "definition": description,
                        "id": identifier,
                        "label": label
                    })

    if definitions:
        # Normalize the output to ensure it is a flat list of dictionaries
        normalized_definitions = []
        for definition in definitions:
            def_text = definition['definition']
            def_id = definition['id']
            normalized_definitions.append({
                "definition": def_text,
                "source": endpoint['name'],
                "id": def_id,
                "label": label
            })
        logger.debug(f'Extracted definitions for {term}: {normalized_definitions}')
        return normalized_definitions
    logger.debug(f"No definitions for {term} available from {endpoint['name']}")
    return None

@audit_logger(functions_to_audit)
def fetch_nodes(tx, node_labels_and_keys, override_query=None, add_labels=True):
    nodes_list = []
    if override_query:
        node_queries  = [override_query]
    else:
        node_queries = []
        for label_key in node_labels_and_keys.keys():
            node_query = f"""
            MATCH (n:{label_key})
            RETURN n
            ORDER BY rand() 
            """
            node_queries.append(node_query)

    for label_key in node_labels_and_keys.keys():
        for node_query in node_queries:
            logger.critical(f"Running query: {node_query}")
            result = tx.run(node_query)
            records = result.data()
            for record in records:
                if add_labels:
                    nodes_list.append((label_key, record['n']))
                else:
                    nodes_list.append(record['n'])
    return nodes_list

@audit_logger(functions_to_audit)
def fetch_nodes_wrapper(driver, node_labels_and_keys, database, override_query=None):
    logger.debug(f"Override Query: {override_query}")
    return run_transaction(driver, fetch_nodes, database, node_labels_and_keys, override_query)

@audit_logger(functions_to_audit)
def add_mentions_to_node(node, field, mentions):
    key = f'{field}_mentions'
    if key not in node:
        node[key] = []
    node[key].extend(mentions)
    logger.debug(f"Added to {key}: {len(mentions)}")

@audit_logger(functions_to_audit)
def load_terms( file_path=None):
    if not file_path:
        file_path = "D:\\Datasets\\capstone\\all_terms.csv"
    logger.info(f"Loading terms from {file_path}")
    with open(file_path, 'r') as file:
        terms = list()
        for line in file.readlines():
            term = line.strip()
            terms.append(term)
        logger.info(f"last term: {term}")
    return terms

@audit_logger(functions_to_audit)
def find_mentions_and_context(text, terms, endpoints, lodc_data_path, window=50, definitions_dict_path=None):
    if definitions_dict_path is None:
        definitions_dict_path = os.path.join(lodc_data_path, 'definitions_dict.json')
    
    if os.path.exists(definitions_dict_path):
        with open(definitions_dict_path, 'r') as json_file:
            definitions_dict = json.load(json_file)
    else:
        logger.warning('No definitions dict available, attempting to re-create')
        create_definitions_dictionary(terms, endpoints, lodc_data_path)
        if os.path.exists(definitions_dict_path):
            with open(definitions_dict_path, 'r') as json_file:
                definitions_dict = json.load(json_file)
        else:
            logger.error("Failed to create definitions dictionary")
            return []

    filtered_terms = [term for term in terms if term in definitions_dict]

    mentions = []
    cleaned_terms = sorted({t.strip().lower() for t in filtered_terms if t}, key=len, reverse=True)
    if cleaned_terms:
        logger.info(f"Processing {len(cleaned_terms)} terms")
    else: 
        logger.warning(f"No terms to process")
        return []

    for term in cleaned_terms:
        matches = list(re.finditer(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
        if matches: 
            logger.debug(f"Found {len(matches)} match(es) for term '{term}'")
        else:
            # logger.debug(f"No matches for term '{term}'") # Overwhelming majority of cases
            continue

        for match in matches:
            start_pos = match.start()
            context_start = max(0, start_pos - window)
            context_end = min(len(text), start_pos + len(term) + window)
            context = text[context_start:context_end]
            mention = {'term': text[start_pos:start_pos+len(term)], 'start_pos': start_pos, 'context': context}
            mentions.append(mention)

    if mentions:
        logger.info(f"found {len(mentions)} mentions")
    else:
        logger.warning(f"No mentions found for any term in text: '{text}'")
        
    return mentions


@audit_logger(functions_to_audit)
def process_endpoints(endpoints, unique_terms, data_path, get_requests, visited_urls, use_ontology=False):
    """
    Processes each endpoint to fetch and cache JSON data based on unique terms.

    Args:
        endpoints (list of dicts): Information about each endpoint to process.
        unique_terms (list of str): Unique terms to check against each endpoint.
        data_path (str): Base directory path for saving JSON files.
        get_requests (dict): Current state of GET requests for managing API calls.
        visited_urls (set): A set of already visited URLs to avoid repeated processing.
    """
    for endpoint in endpoints:
        logger.debug(f'endpoint: {type(endpoint)}')
        endpoint_name = endpoint['name']
        
        logger.debug(f"endpoint['name'] type: {type(endpoint_name)}")
        logger.debug(f"Using endpoint '{endpoint['name']}'")
        
        for term in unique_terms:
            logger.info(f"Checking '{term}' against endpoint '{endpoint['name']}'")
            term_lower = term.lower()
            term_lower_fname = term_lower.replace(' ', '_')
            directory_path = os.path.join(data_path, endpoint['name'])
            file_path = os.path.join(directory_path, f"{term_lower_fname}.json")

            primary_json, get_requests = get_primary_json(endpoint, term_lower, file_path, get_requests)
            if primary_json and 'locator_path' in endpoint:
                urls = fetch_urls(primary_json, endpoint)
                if urls:
                    logger.debug(f"Fetching secondary JSON from urls: {urls}")
                    secondary_jsons = list()
                    for url in urls:
                        secondary_json, _ = get_secondary_json(url, endpoint, get_requests, visited_urls, use_ontology)
                        if secondary_json:
                            secondary_jsons.append(secondary_json)
                    nest_secondary_json(primary_json, secondary_jsons, endpoint['locator_path'])
            
            if primary_json:
                logger.info(f"Saving processed JSON")
                save_json_response(endpoint, term, primary_json, data_path)

@audit_logger(functions_to_audit)
def get_primary_json(endpoint, term_lower, file_path=None, get_requests=0):
    """
    Fetches the primary JSON data, either from cache or by making a new request.

    Args:
        endpoint (dict): Information about the endpoint to process.
        term_lower (str): The term to check against the endpoint.
        file_path (str): The file path for saving JSON files.
        get_requests (dict): Current state of GET requests for managing API calls.

    Returns:
        tuple: The primary JSON data and the updated get_requests dictionary.
    """
    if file_path:
        if os.path.exists(file_path) and not endpoint['refresh']:
            logger.debug(f"Using cached data for term '{term_lower}'.")
            primary_json = load_json(file_path)
    else:
        logger.info(f"Fetching new data for term '{term_lower}'.")
        primary_json, get_requests = fetch_and_cache_json(endpoint, term_lower, get_requests)
        logger.debug(primary_json)

    return primary_json, get_requests

@audit_logger(functions_to_audit)
def get_secondary_json(url, endpoint, get_requests=0, visited_urls=None, use_ontology=False):
    """
    Fetches the secondary JSON data, considering ontology if applicable.

    Args:
        url (str): The URL to fetch the secondary JSON from.
        endpoint (dict): Information about the endpoint to process.
        get_requests (dict): Current state of GET requests for managing API calls.
        visited_urls (set): A set of already visited URLs to avoid repeated processing.
        use_ontology (bool): Whether to use ontology processing.

    Returns:
        dict: The secondary JSON data.
    """
    endpoint_data_path = endpoint['data_path']
    
    if 'ontology_path' in endpoint and 'ontology_base_url' in endpoint and use_ontology:
        logger.debug(f"Attempting to fetch secondary JSON with ontology. Using path {endpoint_data_path}")
        ontology_base_url = endpoint['ontology_base_url']
        ontology_path = endpoint['ontology_path']
        if ontology_base_url and ontology_path:
            if url.startswith(ontology_base_url):
                logger.info(f"Attempting to process ontology using {url}")
                secondary_json = handle_ontology_url(url, get_requests, ontology_path)
            else:
                logger.debug('Recognized ontology base url and path, but the supplied url does not conform.')
                logger.debug(f"ontology_base_url: {ontology_base_url}")
                logger.debug(f"supplied url: {url}")
                secondary_json = None
        else:
            secondary_json = fetch_json_data(url, endpoint_data_path, visited_urls, get_requests)
    else:
        logger.debug(f"Attempting to fetch secondary JSON without ontology. Using path {endpoint_data_path}")
        secondary_json = fetch_json_data(url, endpoint_data_path, visited_urls, get_requests)
    
    remove_duplicates_from_json(secondary_json)

    return secondary_json, get_requests

@audit_logger(functions_to_audit)
def fetch_and_cache_json(endpoint, term, get_requests):
    """
    Fetches JSON data from a specified URL constructed based on the term and endpoint configuration. The data is retrieved using
    a GET request, and the request's parameters are built dynamically to include any optional settings specified in the endpoint.

    Args:
        endpoint (dict): A dictionary containing the endpoint configuration such as URL, query template, and optional parameters.
        term (str): The search term used to format the query template.
        get_requests (function): A function to execute the GET request. It should accept a URL and headers, and return a response object.

    Returns:
        dict or None: Returns the combined JSON data as a dictionary if the request is successful (HTTP 200); otherwise, returns None.
    """
    # Generate all casing combinations for the term
    term_combinations = generate_casing_combinations(term)
    combined_results = []
    logger.info(f"Using {len(term_combinations)} casing combinations for {term}")
    
    for term_comb in term_combinations:
        # Construct query URL
        query = endpoint['query_template'].format(term=term_comb)
        url = endpoint['url']
        headers = endpoint.get('headers', {})
        
        # Log initial query attempt
        logger.debug(f"Querying {endpoint['name']} with term '{term_comb}' at '{url}'")
        logger.debug(f"query: {query}")
        
        # Set up parameters including any optional ones present in the endpoint configuration
        params = {
            'query': query,
            **{k: endpoint[k] for k in [
                'default-graph-uri', 'format', 'timeout'
            ] if k in endpoint}
        }
        
        # URL encode the parameters
        encoded_params = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        full_url = f"{url}?{encoded_params}"

        # Log the complete URL to be requested
        logger.info(f"Complete URL for request: {full_url}")

        # Perform the GET request using the provided get_request function
        try:
            response, _ = get_request(full_url, get_requests, headers=headers)
            if response.status_code == 200:
                logger.info("Successfully retrieved JSON data.")
                json_data = response.json()
                logger.debug(json_data)
                combined_results.extend(json_data.get('results', {}).get('bindings', []))
            else:
                logger.warning(f"HTTP {response.status_code} received for URL: {full_url}")
        except RuntimeError as e:
            logger.error(f"Runtime error during fetching data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during fetching data: {e}")
            raise

    if combined_results:
        combined_json = {"results": {"bindings": combined_results}}
        return combined_json, get_requests
    else:
        return None, get_requests

@audit_logger(functions_to_audit)
def generate_casing_combinations(term):
    words = term.split(' ')
    casing_combinations = []

    for word in words:
        # Generate combinations of lower and upper case for the first letter
        word_combinations = list(map(''.join, itertools.product(*((c.upper(), c.lower()) for c in word[0]))))
        # Add the rest of the word unchanged
        word_combinations = [w + word[1:] for w in word_combinations]
        casing_combinations.append(word_combinations)

    # Generate all possible combinations of the words
    return [' '.join(comb) for comb in itertools.product(*casing_combinations)]

@audit_logger(functions_to_audit)
def unpack_sublists(lst):
    if isinstance(lst, list) and len(lst) == 1 and isinstance(lst[0], list):
        return unpack_sublists(lst[0])
    return [item[0] if isinstance(item, list) and len(item) == 1 else item for item in lst]

@audit_logger(functions_to_audit)
def search_json_for_keys(data, compound_key):
    """
    Recursively searches the JSON structure for the specified keys.

    Parameters:
    data (dict or list): The JSON structure to search through.
    compound_key (str): The compound key to search for.

    Returns:
    list: The list of found values for the specified keys.
    """
    results = []

    i = 0
    key = None
    while not key: # Handle arbitrary data structures, shouldn't be necessary long-term
        logger.debug(f"Current compound_key: {compound_key}")
        i += 1
        while len(compound_key) == 1:
            if compound_key[0] == []:          
                logger.debug(f"No key found, returning data")
                return data
            else:
                logger.debug(f"Un-nesting compound key")
                compound_key = compound_key[0]
        if i >= 5:
            logger.error(f"Max iterations reached")
            raise
        if isinstance(compound_key, str):
            logger.debug(f"Splitting compound key")
            try: key, rest_keys = compound_key.split('|', 1)
            except: 
                logger.debug(f"Unable to split compound key '{compound_key}', assuming it is atomic")
                key = compound_key
                rest_keys = None
            logger.debug(f"key: {key}, rest_keys: {rest_keys}")
        elif isinstance(compound_key, list) and all(isinstance(key, str) for key in compound_key):
            logger.debug(f"Using compound key as is")
            key = compound_key
        else:
            logger.error(f"Potentially invalid key: {compound_key}")
            raise
    
    logger.critical(f'Searching key: {key} in data: {data}')

    if isinstance(data, dict):
        if key in data:
            new_data = data[key]
            if rest_keys:
                logger.debug(f"rest_keys: {rest_keys}")
                result = search_json_for_keys(new_data, rest_keys)
            else:
                result = new_data
            if result:
                logger.debug(f"New result from direct key search with '{key}': {result}")
                results.append(result)
        elif key == '*':
            for k, v in data.items():
                if rest_keys:
                    logger.debug(f"rest_keys: {rest_keys}")
                    result = search_json_for_keys(v, rest_keys)
                else:
                    result = v
                if result:
                    logger.debug(f"new result from data dict key {k}: {result}")
                    results.append(result)
    elif isinstance(data, list):
        for item in data:
            result = search_json_for_keys(item, compound_key)
            if result:
                logger.debug(f"new result from data list item: {result}")
                results.append(result)
    else:
        if not rest_keys:
            results.append(data)
    try:    results_unpacked = unpack_sublists(results)
    except: 
        logger.debug(f"Unable to unpack results: {results}")
        results_unpacked = None
    return results_unpacked if results_unpacked else None


@audit_logger(functions_to_audit)
def remove_duplicates_from_json(data):
    """
    Removes duplicate sibling subtrees and keys with empty list values from the JSON structure.

    Parameters:
    data (dict or list): The JSON structure to process.

    Returns:
    dict or list: The JSON structure without duplicates and empty lists.
    """
    def make_hashable(item):
        """
        Helper function to convert a nested structure into a hashable form.
        """
        if isinstance(item, dict):
            return tuple((k, make_hashable(v)) for k, v in sorted(item.items()))
        elif isinstance(item, list):
            return tuple(make_hashable(i) for i in item)
        return item

    if isinstance(data, dict):
        # Remove keys with empty list values
        keys_to_remove = [key for key, value in data.items() if isinstance(value, list) and not value]
        for key in keys_to_remove:
            del data[key]
        
        # Remove duplicate sibling subtrees
        items = list(data.items())
        seen = set()
        unique_dict = {}
        for key, value in items:
            hashable_value = make_hashable(value)
            if hashable_value not in seen:
                seen.add(hashable_value)
                unique_dict[key] = value
        
        # Recursively process nested structures
        for key in unique_dict:
            unique_dict[key] = remove_duplicates_from_json(unique_dict[key])
        
        return unique_dict

    elif isinstance(data, list):
        seen = set()
        unique_list = []
        for item in data:
            hashable_item = make_hashable(item)
            if hashable_item not in seen:
                seen.add(hashable_item)
                unique_list.append(remove_duplicates_from_json(item))
        return unique_list

    # For literals or any other type, just return as is
    return data

@audit_logger(functions_to_audit)
def remove_empty(data):
    if isinstance(data, dict):
        for key in list(data.keys()):
            if isinstance(data[key], dict):
                remove_empty(data[key])
                if not data[key]:
                    del data[key]
            elif isinstance(data[key], list):
                data[key] = [v for v in data[key] if v]
                if not data[key]:
                    del data[key]
            elif data[key] is None:
                del data[key]
    elif isinstance(data, list):
        for item in data:
            remove_empty(item)

@audit_logger(functions_to_audit)
def apply_filters(data, key_specific_filters_remove, key_specific_filters_retain):
    def apply_filter_to_list(data_list, filter_conditions, retain):
        filtered_list = []
        for item in data_list:
            match = all(item.get(cond_key) in cond_values for cond_key, cond_values in filter_conditions.items())
            if match and retain:
                filtered_list.append(item)
            elif not match and not retain:
                filtered_list.append(item)
            else:
                logging.info(f"Removing item: {item}")
        return filtered_list

    def recursive_apply(data, keys, filter_conditions, retain):
        if len(keys) == 1:
            if keys[0] == '*':
                for k, v in list(data.items()):
                    if isinstance(v, list):
                        data[k] = apply_filter_to_list(v, filter_conditions, retain)
                    elif isinstance(v, dict):
                        for condition_key, condition_values in filter_conditions.items():
                            if retain:
                                if condition_key in v and v[condition_key] not in condition_values:
                                    del data[k]
                            else:
                                if condition_key in v and v[condition_key] in condition_values:
                                    del data[k]
            elif keys[0] in data:
                if isinstance(data[keys[0]], list):
                    data[keys[0]] = apply_filter_to_list(data[keys[0]], filter_conditions, retain)
                elif isinstance(data[keys[0]], dict):
                    if retain:
                        for condition_key, condition_values in filter_conditions.items():
                            if condition_key in data[keys[0]] and data[keys[0]][condition_key] not in condition_values:
                                del data[keys[0]]
                    else:
                        for condition_key, condition_values in filter_conditions.items():
                            if condition_key in data[keys[0]] and data[keys[0]][condition_key] in condition_values:
                                del data[keys[0]]
                else:
                    if not retain:
                        del data[keys[0]]
        else:
            key = keys[0]
            if key == '*':
                for k, v in list(data.items()):
                    if isinstance(v, dict):
                        recursive_apply(v, keys[1:], filter_conditions, retain)
            elif key in data:
                if isinstance(data[key], dict):
                    recursive_apply(data[key], keys[1:], filter_conditions, retain)
                elif isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            recursive_apply(item, keys[1:], filter_conditions, retain)

    if isinstance(data, dict):
        for filter_key, filter_conditions in key_specific_filters_remove:
            keys = filter_key.split('|')
            recursive_apply(data, keys, filter_conditions, retain=False)
        for filter_key, filter_conditions in key_specific_filters_retain:
            keys = filter_key.split('|')
            recursive_apply(data, keys, filter_conditions, retain=True)

    remove_empty(data)

    return data



@audit_logger(functions_to_audit)
def filter_and_deduplicate_json(data, ksf_rem, ksf_ret):
    """
    Recursively filters and removes duplicates from the JSON structure, applying key-specific filters where specified.

    Parameters:
    data (dict or list): The JSON structure to filter and deduplicate.
    key_specific_filters_remove (list): List of tuples where each tuple contains a key and a dictionary of sibling key-value pairs to filter.
    key_specific_filters_retain (list):
    Returns:
    dict or list: The filtered and deduplicated JSON structure.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            filtered_value = filter_and_deduplicate_json(v, ksf_rem, ksf_ret)
            if filtered_value is not None:
                new_dict[k] = filtered_value
        return remove_duplicates_from_json(apply_filters(new_dict, ksf_rem, ksf_ret))
    elif isinstance(data, list):
        new_list = [filter_and_deduplicate_json(item, ksf_rem, ksf_ret) for item in data if filter_and_deduplicate_json(item, ksf_rem, ksf_ret)]
        return remove_duplicates_from_json(new_list) if new_list else None
    return data

@audit_logger(functions_to_audit)
def search_json(data, target_keys, values):
    if isinstance(data, dict):
        for key, value in data.items():
            if any(target_key.startswith(key) for target_key in target_keys):
                for target_key in target_keys:
                    parent_key, _, child_key = target_key.partition('|')
                    if key == parent_key and isinstance(value, dict) and child_key in value:
                        values.append(value[child_key])
            else:
                search_json(value, target_keys, values)
    elif isinstance(data, list):
        for item in data:
            search_json(item, target_keys, values)

@audit_logger(functions_to_audit)
def collect_values_with_key_specific_filters(json_data, target_keys):
    """
    Recursively collects all values associated with the target keys in the JSON structure.

    Parameters:
    json_data (dict): The JSON structure to search through.
    target_keys (list): The keys whose values we want to collect.

    Returns:
    list: A list of all values associated with the target keys.
    """
    values = []

    search_json(json_data, target_keys, values)

    return values

@audit_logger(functions_to_audit)
def filter_json_by_keys(json_data, target_keys, ksf_rem, ksf_ret):
    """
    Recursively filters the JSON structure to retain only the target keys and the intermediates connecting them to the root.
    Additionally filters sub-dictionaries of specified keys based on key-value pairs.
    Removes duplicate sibling subtrees and keys with empty list values.

    Parameters:
    json_data (dict): The JSON structure to filter.
    target_keys (list): The keys to retain in the filtered JSON.
    key_specific_filters (dict): The keys and their sibling key-value pairs to filter.

    Returns:
    dict: A filtered JSON structure containing only the target keys and intermediates.
    """
    filtered_json = {}
    for compound_key in target_keys:
        keys = compound_key.split('|')
        result = search_json_for_keys(json_data, keys)
        if result:
            filtered_json.update(result)

    return filter_and_deduplicate_json(filtered_json, ksf_rem, ksf_ret)

@audit_logger(functions_to_audit)
def filter_mentions_by_longest_term(mentions, endpoints, lodc_data_path, tokenizer, model, similarity_threshold=0.2):
    """
    Filter mentions to keep only the longest term at each start position with at least one valid definition.

    Args:
        mentions (list): List of mention dictionaries, each containing 'term' and 'start_pos'.
        endpoints (list): List of endpoints for fetching candidate definitions.
        lodc_data_path (str): Path to LODC data.
        tokenizer (object): Tokenizer for encoding text.
        model (object): Model for generating embeddings.
        similarity_threshold (float): Threshold for discarding low-similarity definitions.

    Returns:
        list: A list of filtered mention dictionaries, keeping only the longest term at each start position with valid definitions.

    Raises:
        Exception: If the input `mentions` is not a list or if any mention is not a dictionary.
    """
    longest_mentions = {}
    if not isinstance(mentions, list):
        logger.error(f"mentions: Expected list, got {type(mentions)}")
        raise
    logger.debug(f"Checking {len(mentions)} mentions")

    for mention in mentions:
        if not isinstance(mention, dict):
            logger.error(f"mentions: Expected dict, got {type(mentions)}")
            raise Exception("Each mention should be a dictionary")
        
        term = mention['term']
        context = mention['context']
        start_pos = mention['start_pos']
        
        # Generate the context embedding for the mention
        _, context_embedding = encode_context((term, start_pos), context, tokenizer, model)
        mention['context_embedding'] = context_embedding

    for mention in mentions:
        if not isinstance(mention, dict):
            logger.error(f"mentions: Expected dict, got {type(mentions)}")
            raise
        start_pos = mention['start_pos']
        term = mention['term']

        logger.debug(f"Checking term '{term}'")

        # Check if the mention has valid definitions
        logger.info(f"calling process_candidate_definitions with mention {type(mention)}, len {len(mention)}")
        candidate_definitions = process_candidate_definitions(
            mention, endpoints, lodc_data_path, tokenizer, model
        )

        logger.info(f"Got {len(candidate_definitions)} candidate definitions: {type(candidate_definitions)}")
        logger.debug(f"candidate_definitions keys: {candidate_definitions.keys()}")
        logger.debug(f'mention keys: {mention.keys()}')
        logger.debug(f"Calling rank_embeddings with {len(candidate_definitions)} candidate definitions")
        term_result = rank_embeddings(candidate_definitions, term, mention['context_embedding'], start_pos)

        logger.info(f"Got term_result: {term_result}") # Should we be ranking definitions here?

        if 'sources' in term_result and term_result['sources']:
            if start_pos not in longest_mentions:
                longest_mentions[start_pos] = mention
            elif len(term) > len(longest_mentions[start_pos]['term']):
                longest_mentions[start_pos] = mention

    logger.info(f"Longest mentions: {len(longest_mentions.values())}")
    for mention in longest_mentions.values():
        logger.debug(f"{mention['term']}, {mention['start_pos']}")
    
    return list(longest_mentions.values())

@audit_logger(functions_to_audit)
def process_candidate_definitions(mention, endpoints, lodc_data_path, tokenizer, model):
    logger.debug(f"mention :\n{format_json_topology(mention)}") # At this point, mention includes only information from the text

    term = mention['term']
    candidate_definitions = {}

    logger.debug(f"term: '{term}', start_pos: '{mention['start_pos']}', context:'{mention['context']}', embedding: {type(mention['context_embedding'])} {len(mention['context_embedding'])}")

    mention_serializable = convert_to_serializable(mention)  # context contains a tensor
    mention_text = json.dumps(mention_serializable)  # Use the entire row as mention text
    mention_hash = hashlib.md5(mention_text.encode('utf-8')).hexdigest()

    if term in candidate_definitions:
        logger.warning(f"'{term}' already in candidate_definitions")
    else:
        candidate_definitions[term] = {}
        for endpoint in endpoints:
            logger.debug(f"Checking endpoint {endpoint['name']}") 
            definitions = load_and_extract_definitions(term, endpoint, lodc_data_path)
            if definitions:
                logger.debug(f"Got definitions:\n{format_json_topology(definitions)} ")
                if mention_hash not in candidate_definitions[term]:
                    candidate_definitions[term][mention_hash] = []

                for definition_dict in definitions:
                    logger.debug(f'pcd definition_dict:\n{format_json_topology(definition_dict)}')
                    source = definition_dict['source']
                    definition_text = definition_dict['definition']
                    label = definition_dict['label']

                    if definition_text.strip() in ['Wikimedia disambiguation page']:
                        logger.info(f"Skipping meta page for {term} from {source}")
                        continue

                    if definition_text.lower().strip() == term.lower().strip():
                        logger.info(f"Skipping trivial definition for {term} from {source}")
                        continue

                    definition_text = jmespath.search('definition', definition_dict)
                    logger.debug(f'term: {term}, definition text: {definition_text}')
                    
                    embedding = get_embedding(definition_text, tokenizer, model)
                    if embedding is None:
                        logger.error(f'Failed to get embedding for {definition_text}')
                        raise ValueError("Failed to get embedding")
                    
                    definition_dict['embedding'] = embedding
                    if source == 'DBPedia':
                        logger.info(f"definition_dict:\n{format_json_topology(definition_dict)}")
                        logger.info(f"""
                            definition: {definition_text}
                            source: {source}
                            id: {definition_dict['id']}
                            label: {label}""")
                    candidate_definitions[term][mention_hash].append(definition_dict)
                    logger.debug(f"definition text: {definition_dict['definition']}")
                logger.debug(f"definitions:\n{format_json_topology(definitions)}")
            else:
                logger.info(f"No definitions for {term} from {endpoint['name']}")        

    # Normalize the candidate definitions
    logger.debug(f"Normalizing these candidate definitions:\n{format_json_topology(candidate_definitions)}")
    normalized_candidate_definitions = normalize_candidate_definitions(candidate_definitions)
    logger.debug(f"Got normalized_candidate_definitions.\nDoes it contain a proper LODC URL identifier?\n{format_json_topology(normalized_candidate_definitions)}")
    return normalized_candidate_definitions

audit_logger(functions_to_audit)
def create_definitions_dictionary(terms, endpoints, lodc_data_path, output_path, check_file_content_length=True):
    definitions_dict = {}
    logger.debug(f'Processing {len(terms)} terms')

    for i, term in enumerate(terms):
        valid_endpoints = []
        for endpoint in endpoints:
            term_lower_fpath = term.lower().replace(' ', '_')
            intermediate_str = f"{endpoint['name']}\\{term_lower_fpath}.json"
            file_path = os.path.join(lodc_data_path, intermediate_str)

            if os.path.exists(file_path):
                if check_file_content_length:
                    if os.path.getsize(file_path) > 4:
                        valid_endpoints.append(endpoint['name'])
                        break
                else:
                    try:
                        json_data = load_json(file_path)
                        if json_data:
                            valid_endpoints.append(endpoint['name'])
                            break
                    except Exception as e:
                        logger.error(f'Failed to load or parse JSON file: {str(e)}')
        
        if valid_endpoints:
            definitions_dict[term] = valid_endpoints
        if i % 1000 == 0: logger.info(f"Processed {i} terms")

    with open(output_path, 'w') as json_file:
        json.dump(definitions_dict, json_file, indent=4)
    logger.info(f"Definitions dictionary saved to {output_path}")

@audit_logger(functions_to_audit)
def fetch_and_process_nodes(driver, node_labels_and_keys, terms, tokenizer, model, endpoints, lodc_data_path, window=50, 
                            database='rmf2lod2stig', max_nodes=1, dry_run=True, similarity_threshold=0.2):
    nodes_list = fetch_nodes_wrapper(driver, node_labels_and_keys, database=database)
    with driver.session(database=database) as session:
        return process_nodes(nodes_list, node_labels_and_keys, terms, tokenizer, model, endpoints, 
                      lodc_data_path, window, session, max_nodes, dry_run, similarity_threshold)
    
        

@audit_logger(functions_to_audit)
def process_nodes(nodes_list, node_labels_and_keys, terms, tokenizer, model, endpoints, 
                  lodc_data_path, window, session, max_nodes, dry_run=True, similarity_threshold=0.2,
                  override_query=None):
    node_count = 0
    rdf_queries = 0
    merge_queries = 0

    for i, (node_label, node) in enumerate(nodes_list):
        logger.debug(f'checking node: {node}')
        process_node_fields(node, node_label, node_labels_and_keys, terms, tokenizer, model, endpoints, lodc_data_path)
        num_defs_added = add_best_lodc_definitions(node, node_label, node_labels_and_keys, endpoints, 
                                                   lodc_data_path, tokenizer, model, terms, window=window,
                                                   similarity_threshold=similarity_threshold)
        node_count += 1
        if num_defs_added > 0:
            rdf_queries, merge_queries = generate_and_execute_queries(
                node, node_label, node_labels_and_keys, endpoints, window, session, 
                rdf_queries, merge_queries, similarity_threshold, dry_run=dry_run) 
            # Similarity_threshold is used for belt-and-suspenders checking.
            # Later versions should factor this out.
        else: logger.info(f"No definitions added for this node")

        if node_count % 1 == 0:
            logger.info(f"nodes: {node_count}, rdf queries: {rdf_queries}, merge queries: {merge_queries}")
        if i >= max_nodes - 1:
            break
    return node_count, rdf_queries, merge_queries

@audit_logger(functions_to_audit)
def process_node_fields(node, node_label, node_labels_and_keys, terms, tokenizer, model, endpoints, lodc_data_path):
    total_mentions = 0
    for field in node_labels_and_keys[node_label]['fields']:
        if field in node:
            logger.debug(f"Checking for mentions in field '{field}'")
            mentions = find_mentions_and_context(node[field], terms, endpoints, lodc_data_path, window=50)

            if not mentions: 
                logger.debug(f"No mentions of known terms in {field}: {node[field]}")
            else: 
                logger.debug(f"Found {len(mentions)} term mentions in {field}")
            total_mentions += len(mentions)
            for mention in mentions:
                logger.debug(f'mention: {mention}')
                term, context_embedding = encode_context((mention['term'], mention['start_pos']), mention['context'], tokenizer, model)
                logger.debug(f"term: {term}, context: {mention['context']}")
                mention['context_embedding'] = context_embedding
                add_mentions_to_node(node, field, [mention])
        else: 
            logger.warning(f"field {field} not found in node")
    logger.debug(f"Found {total_mentions} total mentions in node")

@audit_logger(functions_to_audit)
def generate_and_execute_queries(
    node, node_label, node_labels_and_keys, endpoints, window, 
    session, rdf_queries, merge_queries, similarity_threshold, dry_run=True):
    for field in node_labels_and_keys[node_label]['fields']:
        mentions_key = f'{field}_mentions'
        if mentions_key in node:
            for mention in node[mentions_key]:
                rdf_query, merge_query = create_query_from_mention(
                    mention, node, node_label, field, endpoints, window, node_labels_and_keys, similarity_threshold)

                if rdf_query:
                    if not dry_run:
                        result = run_query(session, rdf_query)
                        if isinstance(result, dict) and 'error' in result:
                            logger.error(f"Error running rdf_query: {result['error']}")
                            logger.debug(result['traceback'])
                        else:
                            logger.debug(f"Running query")
                        rdf_queries += 1

                if merge_query:
                    if not dry_run:        
                        result = run_query(session, merge_query)
                        if isinstance(result, dict) and 'error' in result:
                            logger.error(f"Error running merge_query: {result['error']}")
                            logger.debug(result['traceback'])
                        else:
                            logger.debug(f"Running query")
                        merge_queries += 1
        else: 
            logger.warning(f'mentions_key {mentions_key} not found in node')
    return rdf_queries, merge_queries

@audit_logger(functions_to_audit)
def run_query(session, query, parameters=None, use_transaction=True, write=True, unpack=True):
    """
    Execute a given Cypher query using the provided Neo4j session, optionally within a transaction.
    
    Args:
    session (neo4j.Session): The Neo4j session instance.
    query (str): The Cypher query to be executed.
    parameters (dict, optional): A dictionary of parameters for the query. Defaults to None.
    use_transaction (bool, optional): If True, runs the query within a transaction. Defaults to True.
    write (bool, optional): If True, runs the query as a write transaction. Defaults to True.
    
    Returns:
    list: A list of result records or a dictionary containing error details.
    """
    if parameters is None:
        parameters = {}

    try:
        def transaction_function(tx, parameters):
            result = tx.run(query, parameters)
            return result
        
        if use_transaction:
            if write:
                result = session.write_transaction(transaction_function, parameters)
            else:
                result = session.read_transaction(transaction_function, parameters)
        else:
            result = session.run(query, parameters)
        
        if unpack:
            result = [record.data() for record in result]

        return result

    except Exception as e:
        # Return the error details to be handled by the calling context
        return {'error': str(e), 'traceback': traceback.format_exc()}

@audit_logger(functions_to_audit)
def get_json_topology(json_obj):
    def _explore(obj):
        if isinstance(obj, dict):
            return {k: _explore(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if obj:
                return [_explore(obj[0])]
            else:
                return []
        else:
            return type(obj).__name__

    topology = _explore(json_obj)
    return topology

@audit_logger(functions_to_audit)
def normalize_structure(structure):
    if isinstance(structure, dict):
        return {key: normalize_structure(value) for key, value in structure.items()}
    elif isinstance(structure, list):
        return [normalize_structure(structure[0])] if structure else []
    else:
        return structure

@audit_logger(functions_to_audit)
def remove_keys(structure):
    if isinstance(structure, dict):
        return sorted((remove_keys(v) for v in structure.values()), key=lambda x: str(type(x)))
    elif isinstance(structure, list):
        return [remove_keys(structure[0])] if structure else []
    else:
        return structure

@audit_logger(functions_to_audit)
def compare_json_structures(json1, json2):
    topology1 = get_json_topology(json1)
    topology2 = get_json_topology(json2)

    normalized_topology1 = normalize_structure(topology1)
    normalized_topology2 = normalize_structure(topology2)

    structure1 = remove_keys(normalized_topology1)
    structure2 = remove_keys(normalized_topology2)

    return structure1 == structure2