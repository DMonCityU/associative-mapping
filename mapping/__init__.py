import importlib
import os
import re

def parse_version(version_str):
    """Convert a version string like '1_2' into a tuple of integers (1, 2)."""
    return tuple(map(int, version_str.split('_')))

def version_matches_condition(version, condition):
    """Check if a version tuple matches a given condition (e.g., '== 1_2')."""
    op = condition[:2].strip()
    target_version = parse_version(condition[2:].strip())
    
    if op == '==':
        return version == target_version
    elif op == '!=':
        return version != target_version
    elif op == '<=':
        return version <= target_version
    elif op == '>=':
        return version >= target_version
    elif op == '<':
        return version < target_version
    elif op == '>':
        return version > target_version
    else:
        raise ValueError(f"Invalid condition operator: {op}")

def get_latest_version(module, version_conditions=None):
    pattern = re.compile(rf'{module}_v(\d+(?:_\d+)*)\.py')
    versions = []

    for file in os.listdir(os.path.dirname(__file__)):
        match = pattern.match(file)
        if match:
            version_str = match.group(1)
            versions.append(parse_version(version_str))

    if not versions:
        raise ImportError(f"No versions of the {module} module found")

    if version_conditions:
        valid_versions = [v for v in versions if version_matches_condition(v, version_conditions)]
        if valid_versions:
            return '_'.join(map(str, max(valid_versions)))
        else:
            raise ImportError(f"No valid versions of the {module} module found for condition '{version_conditions}'")
    else:
        return '_'.join(map(str, max(versions)))

def import_specific_version(module, version_condition=None):
    version = get_latest_version(module, version_condition)
    return importlib.import_module(f".{module}_v{version}", package=__name__)
