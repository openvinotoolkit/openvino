// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/low_precision/eltwise_base_transformation.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

bool isBroadcasted(const Shape& shape) noexcept {
    const size_t spatialIndex = shape.size() == 1 ? 0ul : (shape.size() == 2ul ? 1ul : 2ul);
    for (size_t i = spatialIndex; i < shape.size(); ++i) {
        if (shape[i] != 1ul) {
            return false;
        }
    }

    return true;
}

bool isBranchWithTargetType(const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) {
    if (fakeQuantize == nullptr) {
        return false;
    }

    const std::shared_ptr<Node> parent = fakeQuantize->get_input_node_shared_ptr(0);

    const std::deque<descriptor::Output>& parentOutputs = parent->get_outputs();
    if ((parentOutputs.size() != 1ul) || (parentOutputs.begin()->get_inputs().size() != 1ul)) {
        return false;
    }

    return is_type<opset1::Convolution>(parent) || is_type<opset1::GroupConvolution>(parent) || is_type<opset1::MatMul>(parent);
}

bool EltwiseBaseTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    if (operation->get_input_size() != 2ul) {
        return false;
    }

    FakeQuantizeDequantization dequantization1 = pass::low_precision::NetworkHelper::getDequantization(operation, 0ul);
    FakeQuantizeDequantization dequantization2 = pass::low_precision::NetworkHelper::getDequantization(operation, 1ul);
    if (dequantization1.empty() && dequantization2.empty()) {
        return false;
    }

    if (dequantization1.empty() && !is_type<opset1::Constant>(dequantization1.data)) {
        return false;
    }

    if (dequantization2.empty() && !is_type<opset1::Constant>(dequantization2.data)) {
        return false;
    }

    return true;
}

int EltwiseBaseTransformation::getNotEmpty(const std::shared_ptr<Node>& eltwise) const {
    FakeQuantizeDequantization dequantization1 = pass::low_precision::NetworkHelper::getDequantization(eltwise, 0ul);
    if (dequantization1.empty()) {
        return -1;
    }

    FakeQuantizeDequantization dequantization2 = pass::low_precision::NetworkHelper::getDequantization(eltwise, 1ul);
    if (dequantization2.empty()) {
        return -1;
    }

    const std::shared_ptr<opset1::FakeQuantize> fakeQuantize1 = as_type_ptr<opset1::FakeQuantize>(dequantization1.data);
    const std::shared_ptr<opset1::FakeQuantize> fakeQuantize2 = as_type_ptr<opset1::FakeQuantize>(dequantization2.data);

    if (fakeQuantize1 && !fakeQuantize1) {
        return 0;
    }

    if (!fakeQuantize2 && fakeQuantize2) {
        return 1;
    }

    const bool allBranchesAreEqual = isBranchWithTargetType(fakeQuantize1) == isBranchWithTargetType(fakeQuantize2);
    const std::vector<std::shared_ptr<Node>> dataNodes = { dequantization1.data, dequantization2.data };
    for (size_t i = 0; i < dataNodes.size(); ++i) {
        const std::shared_ptr<Node>& data = dataNodes[i];
        if ((allBranchesAreEqual && isBroadcasted(data->get_output_shape(0))) ||
            (!allBranchesAreEqual && isBranchWithTargetType(as_type_ptr<opset1::FakeQuantize>(data)))) {
            return i;
        }
    }

    int fullPathIndex = 0;

    return fullPathIndex;
}
