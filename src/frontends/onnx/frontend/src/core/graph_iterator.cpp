// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/onnx/graph_iterator.hpp"

using namespace ov::frontend::onnx;

GraphIterator::~GraphIterator() = default;

GraphIteratorV2::~GraphIteratorV2() = default;

OPENVINO_SUPPRESS_DEPRECATED_START
std::map<std::string, std::string> GraphIteratorV2::get_metadata() const {
    return get_metadata_ref();
}

std::filesystem::path GraphIteratorV2::get_model_dir() const {
    return get_model_dir_ref();
}
OPENVINO_SUPPRESS_DEPRECATED_END
