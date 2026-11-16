// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "openvino/core/deprecated.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"

namespace ov::frontend::onnx {

// GraphIterator implementations may live in other shared libraries, possibly built against an earlier version of the
// public header or linked against a different C runtime. The data of an iterator is therefore read through
// GraphIteratorV2 (references, copied here with the allocator of this module) whenever the iterator provides it.
// An iterator without GraphIteratorV2 is served by the by-value functions of GraphIterator, as before.
// GraphIterator itself is not touched (no new virtual functions), so these functions never read a vtable slot that an
// older iterator does not have.

/// \brief Returns a copy of the metadata of the iterator.
inline std::map<std::string, std::string> read_graph_metadata(const GraphIterator& iterator) {
    if (const auto* v2 = dynamic_cast<const GraphIteratorV2*>(&iterator)) {
        return v2->get_metadata_ref();
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    return iterator.get_metadata();
    OPENVINO_SUPPRESS_DEPRECATED_END
}

/// \brief Returns a copy of the model directory of the iterator.
inline std::filesystem::path read_graph_model_dir(const GraphIterator& iterator) {
    if (const auto* v2 = dynamic_cast<const GraphIteratorV2*>(&iterator)) {
        return v2->get_model_dir_ref();
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    return iterator.get_model_dir();
    OPENVINO_SUPPRESS_DEPRECATED_END
}

}  // namespace ov::frontend::onnx
