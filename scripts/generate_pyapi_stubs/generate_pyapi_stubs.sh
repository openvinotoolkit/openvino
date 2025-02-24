#!/bin/bash

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

invalid_expressions=(
    # The classes bindings' will be soon added to pyopenvino
    "ov::op::v1::Add"
    "ov::op::v1::Divide"
    "ov::op::v1::Multiply"
    "ov::op::v1::Subtract"
    "ov::op::v1::Divide"

    # Circular dependency Input/Output/Node/Tensor
    "ov::Node"
    "ov::Input<ov::Node>"
    "ov::descriptor::Tensor"
    "<Type: 'undefined'>"
    "ov::Output<ov::Node const>"

    # New bindings required
    "ov::float16"
    "ov::EncryptionCallbacks"
    "ov::streams::Num"

    # Other issues
    "<Dimension:"
)

invalid_identifiers=(
    # Ticket:
    "<locals>"
)

unresolved_names=(
    # Circular dependencies
    "InferRequestWrapper"
    "RemoteTensorWrapper"
    "capsule"
    "VASurfaceTensorWrapper"

    # Other issues
    "_abc._abc_data"
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
output_dir="$(dirname "$0")/../../src/bindings/python/src"

python -m pybind11_stubgen \
            --output-dir "$output_dir" \
            --root-suffix "" \
            --ignore-invalid-expressions "$invalid_expressions_regex" \
            --ignore-invalid-identifiers "$invalid_identifiers_regex" \
            --ignore-unresolved-names "$unresolved_names_regex" \
            --numpy-array-use-type-var \
            --exit-code \
            openvino

# Workaround for pybind11-stubgen issue where it doesn't import some modules for stubs generated from .py files
pyi_file="$output_dir/openvino/_ov_api.pyi"
if [ -f "$pyi_file" ]; then
    sed -i '2i import typing' "$pyi_file"
    sed -i '2i import pathlib' "$pyi_file"
else
    echo "File $pyi_file not found."
fi
