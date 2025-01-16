// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace util {

inline bool is_dynamic_node(const std::shared_ptr<ov::Node>& node) {
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        if (node->get_input_partial_shape(i).is_dynamic()) {
            return true;
        }
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return true;
        }
    }
    return false;
}

inline bool is_dynamic_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (is_dynamic_node(parameter)) {
            return true;
        }
    }
    for (const auto& result : model->get_results()) {
        if (is_dynamic_node(result)) {
            return true;
        }
    }
    return false;
}

}  // namespace util
}  // namespace ov
