// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/prelu.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void PReluTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::PRelu>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

bool PReluTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> prelu = m.get_match_root();
    if (!canBeTransformed(context, prelu)) {
        return false;
    }

    prelu = NetworkHelper::separateInStandaloneBranch(prelu);
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(prelu, 0);
    moveDequantizationAfter(context, prelu, dequantization, false, false);
    return true;
}

bool PReluTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return false;
}

bool PReluTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, 0);
    if (dequantization.empty() || (dequantization.subtract != nullptr)) {
        return false;
    }

    const std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(dequantization.multiply->input_value(1).get_node_shared_ptr());
    const auto scales = constant->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](const float value) { return value < 0.f; })) {
        return false;
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
