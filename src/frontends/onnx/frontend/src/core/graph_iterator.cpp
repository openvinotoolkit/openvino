// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/onnx/graph_iterator.hpp"

using namespace ov::frontend::onnx;

GraphIterator::~GraphIterator() = default;

const std::map<std::string, std::string>& GraphIterator::get_metadata_ref() const {
    std::call_once(m_legacy_metadata_once, [this] {
        OPENVINO_SUPPRESS_DEPRECATED_START
        m_legacy_metadata = get_metadata();
        OPENVINO_SUPPRESS_DEPRECATED_END
    });
    return m_legacy_metadata;
}

const std::filesystem::path& GraphIterator::get_model_dir_ref() const {
    std::call_once(m_legacy_model_dir_once, [this] {
        OPENVINO_SUPPRESS_DEPRECATED_START
        m_legacy_model_dir = get_model_dir();
        OPENVINO_SUPPRESS_DEPRECATED_END
    });
    return m_legacy_model_dir;
}
