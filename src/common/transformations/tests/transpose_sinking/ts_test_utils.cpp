// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ts_test_utils.hpp"

#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;

namespace transpose_sinking {
namespace testing {
namespace utils {

shared_ptr<Node> create_main_node(const OutputVector& inputs, size_t num_ops, const FactoryPtr& creator) {
    OutputVector current_inputs = inputs;
    for (size_t i = 0; i < num_ops; ++i) {
        auto op = creator->create(current_inputs);
        current_inputs[0] = op->output(0);
    }
    return current_inputs[0].get_node_shared_ptr();
}

ParameterVector filter_parameters(const OutputVector& out_vec) {
    ParameterVector parameters;
    for (const auto& out : out_vec) {
        auto node = out.get_node_shared_ptr();
        if (auto param = ov::as_type_ptr<Parameter>(node)) {
            parameters.push_back(param);
        }
    }
    return parameters;
}

OutputVector set_transpose_for(const vector<size_t>& idxs, const OutputVector& out_vec) {
    OutputVector result = out_vec;
    for (const auto& idx : idxs) {
        const auto& out = out_vec[idx];
        auto rank = out.get_partial_shape().rank().get_length();
        vector<int64_t> axes(rank);
        iota(axes.begin(), axes.end(), 0);
        reverse(axes.begin(), axes.end());
        auto order = make_shared<Constant>(element::i32, Shape{axes.size()}, axes);
        auto transpose = make_shared<Transpose>(out, order);
        result[idx] = transpose;
    }
    return result;
}

OutputVector set_transpose_with_order(const vector<size_t>& idxs,
                                      const OutputVector& out_vec,
                                      const vector<size_t>& transpose_order_axes) {
    OutputVector result = out_vec;
    for (const auto& idx : idxs) {
        const auto& out = out_vec[idx];
        auto order = make_shared<Constant>(element::i32, Shape{transpose_order_axes.size()}, transpose_order_axes);
        auto transpose = make_shared<Transpose>(out, order);
        result[idx] = transpose;
    }
    return result;
}

OutputVector set_gather_for(const vector<size_t>& idxs, const OutputVector& out_vec) {
    OutputVector result = out_vec;
    for (const auto& idx : idxs) {
        const auto& out = out_vec[idx];
        vector<int64_t> axes(out.get_shape()[0]);
        iota(axes.begin(), axes.end(), 0);
        reverse(axes.begin(), axes.end());
        auto order = make_shared<Constant>(element::i32, Shape{axes.size()}, axes);
        auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
        auto transpose = make_shared<Gather>(out, order, axis);
        result[idx] = transpose;
    }
    return result;
}

std::string to_string(const Shape& shape) {
    ostringstream result;
    result << "{";
    for (size_t idx = 0; idx < shape.size(); ++idx) {
        if (idx)
            result << ",";
        result << shape[idx];
    }
    result << "}";
    return result.str();
}

Output<Node> parameter(ov::element::Type el_type, const PartialShape& ps) {
    return std::make_shared<Parameter>(el_type, ps);
}

}  // namespace utils
}  // namespace testing
}  // namespace transpose_sinking
