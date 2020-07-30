// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/subtract_multiply_to_multiply_add.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void SubtrcatMultiplyToMultiplyAddTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Multiply>(pass, context);
}

FakeQuantizeDequantization get(const std::shared_ptr<Node> node) {
    std::shared_ptr<Node> dataNode = node;

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = is_type<opset1::Constant>(dataNode->get_input_node_shared_ptr(1)) ?
        as_type_ptr<ngraph::opset1::Multiply>(dataNode) :
        nullptr;
    if (multiply != nullptr) {
        dataNode = multiply->get_input_node_shared_ptr(0);
    }

    const std::shared_ptr<opset1::Subtract> subtract = (dataNode->get_input_size() > 1ul) && is_type<opset1::Constant>(dataNode->get_input_node_ptr(1)) ?
        as_type_ptr<opset1::Subtract>(dataNode) :
        nullptr;
    if (subtract != nullptr) {
        dataNode = subtract->get_input_node_shared_ptr(0);
    }

    const std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(dataNode);
    if (convert != nullptr) {
        dataNode = convert->get_input_node_shared_ptr(0);
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, multiply);
}

void SubtrcatMultiplyToMultiplyAddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto multiply = m.get_match_root();
    FakeQuantizeDequantization dequantization = get(multiply);
    if (dequantization.empty()) {
        return;
    }

    multiply = separateInStandaloneBranch(multiply);
    dequantization = get(multiply);
    if (dequantization.empty()) {
        return;
    }

    std::shared_ptr<Node> lastNew = dequantization.data;

    if (dequantization.multiply != nullptr) {
        auto multiplyConstant = dequantization.multiply->get_input_node_shared_ptr(1);
        lastNew = std::make_shared<opset1::Multiply>(lastNew, multiplyConstant);
        NetworkHelper::copyInfo(dequantization.multiply, lastNew);
    }

    if (dequantization.subtract != nullptr) {
        auto subtractConstant = dequantization.subtract->get_input_node_shared_ptr(1);
        lastNew = std::make_shared<opset1::Add>(lastNew, fold<opset1::Multiply>(
            fold<opset1::Multiply>(
                subtractConstant,
                std::make_shared<opset1::Constant>(subtractConstant->get_output_element_type(0), Shape{}, std::vector<float>{ -1.f })),
            dequantization.multiply->get_input_node_shared_ptr(1)));
        NetworkHelper::copyInfo(dequantization.subtract, lastNew);
    }

    const std::shared_ptr<Node> lastOriginal = dequantization.multiply == nullptr ?
        as_type_ptr<Node>(dequantization.subtract) :
        dequantization.multiply;
    replace_node(lastOriginal, lastNew);

    updateOutput(context, dequantization.multiply, lastNew);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
