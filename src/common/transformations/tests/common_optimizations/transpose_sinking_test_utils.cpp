// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/manager.hpp>

#include "transpose_sinking_test_utils.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace ov;
using namespace ov::opset10;

shared_ptr <Node> create_main_node(const OutputVector &inputs, size_t num_ops, const FactoryPtr &creator) {
    OutputVector current_inputs = inputs;
    for (size_t i = 0; i < num_ops; ++i) {
        auto op = creator->create(current_inputs);
        current_inputs[0] = op->output(0);
    }
    return current_inputs[0].get_node_shared_ptr();
}

ParameterVector filter_parameters(const OutputVector &out_vec) {
    ParameterVector parameters;
    for (const auto& out : out_vec) {
        auto node = out.get_node_shared_ptr();
        if (auto param = dynamic_pointer_cast<Parameter>(node)) {
            parameters.push_back(param);
        }
    }
    return parameters;
}

OutputVector set_transpose_for(const vector<size_t>& idxs, const OutputVector& out_vec) {
    OutputVector result = out_vec;
    for (size_t i = 0; i < out_vec.size(); ++i) {
        if (find(idxs.begin(), idxs.end(), i) != idxs.end()) {
            const auto &out = out_vec[i];
            auto rank = out.get_partial_shape().rank().get_length();
            vector<int64_t> axes(rank);
            iota(axes.begin(), axes.end(), 0);
            reverse(axes.begin(), axes.end());
            auto order = make_shared<Constant>(element::i32, Shape{axes.size()}, axes);
            auto transpose = make_shared<Transpose>(out, order);
            result[i] = transpose;
        }
    }
    return result;
}

std::string to_string(const Shape &shape) {
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
