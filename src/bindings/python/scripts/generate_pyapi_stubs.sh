#!/bin/bash

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# When removing xfails remember to remove them from .bat script as well
invalid_expressions=(
    # The classes bindings' will be soon added to pyopenvino. Ticket: 163077
    "ov::op::v1::Add"
    "ov::op::v1::Divide"
    "ov::op::v1::Multiply"
    "ov::op::v1::Subtract"
    "ov::op::v1::Divide"

    # Circular dependency Input/Output/Node/Tensor. Ticket: 163078
    "ov::Node"
    "ov::Input<ov::Node>"
    "ov::descriptor::Tensor"
    "<Type: 'undefined'>"
    "ov::Output<ov::Node const>"

    # New bindings required, ticket: 163094
    "ov::float16"
    "ov::EncryptionCallbacks"
    "ov::streams::Num"
    "ov::pass::pattern::PatternSymbolValue"

    # Other issues, ticket: 163094
    "<Dimension:"
    "dynamic"
)

invalid_identifiers=(
    # Ticket: 163093
    "<locals>"
)

unresolved_names=(
    # Circular dependencies, ticket: 163078
    "InferRequestWrapper"

    # Other issues, ticket: 163094
    "RemoteTensorWrapper"
    "capsule"
    "VASurfaceTensorWrapper"
    "_abc._abc_data"
    "openvino._ov_api.undefined_deprecated"
)

create_regex_pattern() {
    local errors=("$@")
    local regex_pattern=""

    for error in "${errors[@]}"; do
        escaped_error=$(printf '%s\n' "$error" | sed -e 's/[]\/$*.^|[]/\\&/g')
        regex_pattern+=".*$escaped_error.*|"
    done
    regex_pattern="${regex_pattern%|}"
    echo "$regex_pattern"
}

invalid_expressions_regex=$(create_regex_pattern "${invalid_expressions[@]}")
invalid_identifiers_regex=$(create_regex_pattern "${invalid_identifiers[@]}")
unresolved_names_regex=$(create_regex_pattern "${unresolved_names[@]}")

# Set the output directory
if [ -z "$1" ]; then
    output_dir="$(dirname "$0")/../src"
else
    output_dir="$1"
fi

# Generate stubs for C++ bindings
if ! python -m pybind11_stubgen \
            --output-dir "$output_dir" \
            --root-suffix "" \
            --ignore-invalid-expressions "$invalid_expressions_regex" \
            --ignore-invalid-identifiers "$invalid_identifiers_regex" \
            --ignore-unresolved-names "$unresolved_names_regex" \
            --print-invalid-expressions-as-is \
            --numpy-array-use-type-var \
            --exit-code \
            openvino; then
    echo "Error: pybind11-stubgen failed."
    exit 1
fi

# Check if the stubs were actually generated
if [ "$(ls -A "$output_dir/openvino")" ]; then
    echo "Stub files generated successfully."
else
    echo "No stub files were generated."
    exit 1
fi
echo "Python stubs generated."
# Workaround for pybind11-stubgen issue where it doesn't import some modules for stubs generated from .py files 
# Ticket: 163225
pyi_file="$output_dir/openvino/_ov_api.pyi"
if [ -f "$pyi_file" ]; then
    sed -i '2i import typing' "$pyi_file"
    sed -i '2i import pathlib' "$pyi_file"
else
    echo "File $pyi_file not found."
    exit 1
fi
