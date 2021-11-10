// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <openvino/opsets/opset1.hpp>
#include <openvino/core/validation_util.hpp>

template <class T>
inline bool get_data_as_int64(
        size_t idx, const ov::Node* op, std::vector<int64_t>& axes_value,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    if (constant_data.count(idx)) {
        axes_value = ov::opset1::Constant(constant_data.at(idx)).cast_vector<int64_t>();
    } else {
        const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
        axes_value = constant->cast_vector<int64_t>();
    }
    return true;
}

template <>
inline bool get_data_as_int64<ov::PartialShape>(
        size_t idx, const ov::Node* op, std::vector<int64_t>& axes_value,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (constant_data.count(idx)) {
        axes_value = ov::opset1::Constant(constant_data.at(idx)).cast_vector<int64_t>();
    } else if (const auto& constant = ov::get_constant_from_source(op->input_value(idx))) {
        axes_value = constant->cast_vector<int64_t>();
    } else {
        return false;
    }
    return true;
}
