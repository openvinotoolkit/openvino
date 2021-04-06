// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_max.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ReduceMaxTransformation::ReduceMaxTransformation(const Params& params) : ReduceBaseTransformation(params) {}

void ReduceMaxTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::ReduceMax>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

bool ReduceMaxTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    if (!is_type<opset1::ReduceMax>(reduce)) {
        return false;
    }

    if (!ReduceBaseTransformation::canBeTransformed(context, reduce)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(reduce);
    const std::vector<float> scales = as_type_ptr<opset1::Constant>(dequantization.multiplyConstant)->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](const float value) { return value < 0.0; })) {
        return false;
    }

    return true;
}

bool ReduceMaxTransformation::isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept {
    return true;
}

bool ReduceMaxTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
