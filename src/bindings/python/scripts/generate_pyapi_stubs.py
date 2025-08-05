import os
import re
import subprocess
from pathlib import Path
import sys

# Constants
INVALID_EXPRESSIONS = [
    "ov::Node",                               # In openvino._pyopenvino.Input.get_node : Invalid expression 'ov::Node'
    "ov::Input<ov::Node>",                    # In openvino._pyopenvino.Output.get_target_inputs : Invalid expression 'ov::Input<ov::Node>'
    "ov::descriptor::Tensor",                 # In openvino._pyopenvino.Output.get_tensor : Invalid expression 'ov::descriptor::Tensor'
    "ov::Output<ov::Node const>",             # In openvino._pyopenvino.Tensor.__init__ : Invalid expression 'ov::Output<ov::Node const>'
    "ov::float16",                            # In openvino._pyopenvino.op.Constant.__init__ : Invalid expression 'ov::float16'
    "ov::EncryptionCallbacks",                # In openvino._pyopenvino.properties.cache_encryption_callbacks : Invalid expression 'ov::EncryptionCallbacks'
    "ov::streams::Num",                       # In openvino._pyopenvino.properties.num_streams : Invalid expression 'ov::streams::Num'
    "ov::pass::pattern::PatternSymbolValue",  # In openvino._pyopenvino.passes.Predicate.__init__ : Invalid expression 'ov::pass::pattern::PatternSymbolValue'
    "<Dimension:",                            # In openvino._pyopenvino.PartialShape.dynamic : Invalid expression '<Dimension: ?>'
    "<RTMap>",                                # In openvino._pyopenvino.Model.evaluate : Invalid expression '<RTMap>'
    "<Type: 'dynamic'>"                       # In openvino._pyopenvino.Tensor.__init__ : Invalid expression '<Type: 'dynamic'>'
]
INVALID_IDENTIFIERS = ["<locals>"]            # In openvino.properties.* : Invalid identifier '<locals>' at 'openvino.properties.device.__getattr__'
UNRESOLVED_NAMES = [
    "InferRequestWrapper",                    # In openvino._pyopenvino.CompiledModel.create_infer_request : Can't find/import 'InferRequestWrapper'
    "RemoteTensorWrapper",                    # In openvino._pyopenvino.Tensor.copy_* : Can't find/import 'RemoteTensorWrapper'
    "capsule",                                # In openvino._pyopenvino.VAContext.__init__ : Can't find/import 'capsule'
    "VASurfaceTensorWrapper",                 # In openvino._pyopenvino.VAContext.create_tensor : Can't find/import 'VASurfaceTensorWrapper'
    "typing_extensions.CapsuleType",          # In openvino._pyopenvino.VAContext.__init__ : Can't find/import 'typing_extensions.CapsuleType'
    "_abc._abc_data",                         # In openvino.utils.data_helpers.wrappers.OVDict : Can't find/import '_abc._abc_data'
    "openvino._ov_api.undefined_deprecated",  # In openvino._ov_api : Can't find/import 'openvino._ov_api.undefined_deprecated'
    "InputCutInfo",                           # In openvino.tools.ovc.cli_parser : Can't find/import 'InputCutInfo'
    "ParamData"                               # In openvino.tools.ovc.cli_parser : Can't find/import 'ParamData'
]

def create_regex_pattern(errors):
    return "|".join([re.escape(error) for error in errors])

def sanitize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    content = [re.sub(r"<function _get_node_factory at 0x[0-9a-fA-F]+>", "<function _get_node_factory at memory_address>", line) for line in content]
    content = [re.sub(r"__version__: str = '[^']*'", "__version__: str = 'version_string'", line) for line in content]
    content = [re.sub(r"<function <lambda> at 0x[0-9a-fA-F]+>", "<function <lambda> at memory_address>", line) for line in content]
    content = [re.sub(r": \.\.\.", ": typing.Any", line) for line in content]
    content = [re.sub(r"pathlib\._local\.Path", "pathlib.Path", line) for line in content]
    content = [re.sub(r"pass: MatcherPass", "matcher_pass: MatcherPass", line) for line in content]

    # Sort imports
    imports = [line for line in content if line.startswith("from ") or line.startswith("import ")]
    # Separate non-import lines from the content
    non_imports = [
        line for line in content
        if not (line.startswith("from ") or line.startswith("import ") or line.strip() == "# type: ignore")
    ]
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
            if "import typing\n" not in content:
                content.insert(1, "import typing\n")
            if "import pathlib\n" not in content:
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
        sanitize_file(file_path)

if __name__ == "__main__":
    main()
