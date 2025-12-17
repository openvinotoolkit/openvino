import json

db_path = "src/custom_ops/btl/btl_function_database.json"

with open(db_path, 'r') as f:
    data = json.load(f)

# Define Truth Tables
min_tt = [-1, -1, -1, -1, 0, 0, -1, 0, 1]
max_tt = [-1, 0, 1, 0, 0, 1, 1, 1, 1]
neg_tt = [1, 1, 1, 0, 0, 0, -1, -1, -1] # f(a,b) = -a

print("Searching for Universal Functions...")

# Search in NPN Classes
for cls in data.get('npn_classes', []):
    tt = cls['truth_table']
    if tt == min_tt:
        print(f"FOUND MIN in NPN Class {cls['npn_class_id']}, Canonical ID {cls['canonical_function_id']}")
    if tt == max_tt:
        print(f"FOUND MAX in NPN Class {cls['npn_class_id']}, Canonical ID {cls['canonical_function_id']}")
    if tt == neg_tt:
        print(f"FOUND NEG in NPN Class {cls['npn_class_id']}, Canonical ID {cls['canonical_function_id']}")

# Search in Exotic Functions
for name, func in data.get('exotic_functions', {}).items():
    tt = func['truth_table']
    print(f"Exotic Function: {name}, TT: {tt}")
    if tt == min_tt:
        print(f"FOUND MIN in Exotic {name}")
    if tt == max_tt:
        print(f"FOUND MAX in Exotic {name}")
    if tt == neg_tt:
        print(f"FOUND NEG in Exotic {name}")

