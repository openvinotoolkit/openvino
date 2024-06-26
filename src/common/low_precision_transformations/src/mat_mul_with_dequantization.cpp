// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/mat_mul_with_dequantization.hpp"

#include <memory>
#include "low_precision/rt_info/bias_attribute.hpp"

using namespace ov;
using namespace ov::pass;
using namespace ov::pass::low_precision;

MatMulWithDequantizationTransformation::MatMulWithDequantizationTransformation(const Params& params) : MatMulTransformation(params) {
}

void MatMulWithDequantizationTransformation::handleDequantization(const std::shared_ptr<ov::opset1::Multiply>& dequantization) const {
    const auto& dequantization_constant = is_type<opset1::Constant>(dequantization->get_input_node_shared_ptr(1)) ?
        as_type<opset1::Constant>(dequantization->get_input_node_ptr(1)) :
        as_type<opset1::Constant>(dequantization->get_input_node_ptr(0));
    if ((dequantization_constant == nullptr) || (ov::shape_size(dequantization_constant->get_shape()) != 1ull)) {
        return;
    }

    ov::mark_as_bias(dequantization);
}
