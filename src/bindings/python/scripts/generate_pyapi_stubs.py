import os
import re
import subprocess
from pathlib import Path
import sys

# Constants
INVALID_EXPRESSIONS = [
    "ov::op::v1::Add", "ov::op::v1::Divide", "ov::op::v1::Multiply", "ov::op::v1::Subtract", "ov::op::v1::Divide",
    "ov::Node", "ov::Input<ov::Node>", "ov::descriptor::Tensor", "<Type: 'undefined'>", "ov::Output<ov::Node const>",
    "ov::float16", "ov::EncryptionCallbacks", "ov::streams::Num", "ov::pass::pattern::PatternSymbolValue",
    "<Dimension:", "<RTMap>", "<Type: 'dynamic'>"
]
INVALID_IDENTIFIERS = ["<locals>"]
UNRESOLVED_NAMES = [
    "InferRequestWrapper", "RemoteTensorWrapper", "capsule", "VASurfaceTensorWrapper", "_abc._abc_data",
    "openvino._ov_api.undefined_deprecated", "InputCutInfo", "ParamData"
]

def create_regex_pattern(errors):
    return "|".join([re.escape(error) for error in errors])

def sanitize_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    content = [re.sub(r"<function _get_node_factory at 0x[0-9a-fA-F]+>", "<function _get_node_factory at memory_address>", line) for line in content]
    content = [re.sub(r"__version__: str = '[^']*'", "__version__: str = 'version_string'", line) for line in content]
    content = [re.sub(r"<function <lambda> at 0x[0-9a-fA-F]+>", "<function <lambda> at memory_address>", line) for line in content]
    content = [re.sub(r": \.\.\.", ": typing.Any", line) for line in content]
    content = [re.sub(r"pathlib\._local\.Path", "pathlib.Path", line) for line in content]
    content = [re.sub(r"pass: MatcherPass", "matcher_pass: MatcherPass", line) for line in content]

    # Sort imports
    imports = [line for line in content if line.startswith("from ") or line.startswith("import ")]
    non_imports = [line for line in content if not (line.startswith("from ") or line.startswith("import "))]
    sorted_imports = sorted(imports)

    sorted_imports.insert(0, "# type: ignore\n")

    with open(file_path, 'w') as file:
        file.writelines(sorted_imports + non_imports)

def main():
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path(__file__).parent.parent / "src"
    output_dir = output_dir.resolve()

    invalid_expressions_regex = create_regex_pattern(INVALID_EXPRESSIONS)
    invalid_identifiers_regex = create_regex_pattern(INVALID_IDENTIFIERS)
    unresolved_names_regex = create_regex_pattern(UNRESOLVED_NAMES)

    try:
        subprocess.run([
            "python", "-m", "pybind11_stubgen",
            "--output-dir", str(output_dir),
            "--root-suffix", "",
            "--ignore-invalid-expressions", invalid_expressions_regex,
            "--ignore-invalid-identifiers", invalid_identifiers_regex,
            "--ignore-unresolved-names", unresolved_names_regex,
            "--numpy-array-use-type-var",
            "--exit-code",
            "openvino"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: pybind11-stubgen failed with error: {e}")
        exit(1)

    # Workaround for pybind11-stubgen issue where it's missing some imports
    pyi_file = output_dir / "openvino/_ov_api.pyi"
    if pyi_file.exists():
        with open(pyi_file, 'r+') as file:
            content = file.readlines()
            content.insert(1, "import typing\n")
            content.insert(2, "import pathlib\n")
            file.seek(0)
            file.writelines(content)
    else:
        print(f"File {pyi_file} not found.")
        exit(1)

    pyi_file = output_dir / "openvino/tools/ovc/convert_impl.pyi"
    if pyi_file.exists():
        with open(pyi_file, 'r+') as file:
            content = file.readlines()
            content = [line for line in content if not line.startswith("tf_frontend_with_python_bindings_installed: bool")]
            file.seek(0)
            file.writelines(content)
            file.truncate()
    else:
        print(f"File {pyi_file} not found.")
        exit(1)

    # Process each changed .pyi file
    for file_path in output_dir.rglob("*.pyi"):
        print(f"Sanitizing file: {file_path}")
        sanitize_file(file_path)

if __name__ == "__main__":
    main()
