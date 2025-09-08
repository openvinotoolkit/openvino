import os
import json
import requests
import argparse
from jsonschema import Draft7Validator
import sys


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def validate_json_files(benchmarks_dir):
    """Validate all JSON files in the 'data' subdirectory against the schema."""
    # Define the path to the schema file
    schema_path = os.path.join(benchmarks_dir, 'schema', 'graph-data-schema.json')

    # Fetch the schema
    schema = load_json(schema_path)
    validator = Draft7Validator(schema)
    
    # Define the path to the 'data' subdirectory containing the JSON files
    data_dir = os.path.join(benchmarks_dir, 'data')
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    invalid_count = 0  # Track the number of invalid JSON objects

    # Iterate through each JSON file and validate
    for json_file in json_files:
        json_file_path = os.path.join(data_dir, json_file)
        print(f"Validating {json_file_path}...")

        try:
            json_data = load_json(json_file_path)
        except Exception as e:
            print(f"Error loading JSON file {json_file}: {e}")
            invalid_count += 1
            continue

        # Check if the JSON data is a list of objects
        if isinstance(json_data, list):
            # Iterate through each object in the list
            for idx, item in enumerate(json_data):
                errors = list(validator.iter_errors(item))
                if errors:
                    print(f"Validation failed for object {idx} in {json_file}:")
                    for error in errors:
                        print(f"Error: {error.message}")
                    invalid_count += 1
        else:
            # Validate the JSON object itself if it's not a list
            errors = list(validator.iter_errors(json_data))
            if errors:
                print(f"Validation failed for {json_file}:")
                for error in errors:
                    print(f"Error: {error.message}")
                invalid_count += 1

    return invalid_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate JSON files against a schema.")
    parser.add_argument(
        "benchmarks_dir",
        type=str,
        help="Path to the directory containing the benchmark JSON files"
    )
    args = parser.parse_args()
    invalid_jsons = validate_json_files(args.benchmarks_dir)
    
    if invalid_jsons > 0:
        print(f"{invalid_jsons} JSON object(s) are invalid. Failing the build.")
        sys.exit(1)  # Exit with a non-zero status to fail Jenkins/GitHub Actions
    else:
        print("All JSON objects are valid.")
        sys.exit(0)  # Exit with zero status for success

