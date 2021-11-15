// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/prelu.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::PReluTransformation, "PReluTransformation", 0);

PReluTransformation::PReluTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::PRelu>({ pattern::wrap_type<opset1::Multiply>(), pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "PReluTransformation");
    this->register_matcher(m, callback);
}

bool PReluTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
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
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, 0);
    if (dequantization.empty() || (dequantization.subtract != nullptr)) {
        return false;
    }

    const auto scales = dequantization.multiplyConstant->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](const float value) { return value < 0.f; })) {
        return false;
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
