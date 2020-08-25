// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/interpolate.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

void InterpolateTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Interpolate>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

bool InterpolateTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> interpolate = m.get_match_root();
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }
    interpolate = separateInStandaloneBranch(interpolate);
    moveDequantizationAfter(context, interpolate, NetworkHelper::getDequantization(interpolate), true);
    return true;
}

bool InterpolateTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    std::shared_ptr<opset1::Interpolate> interpolate = as_type_ptr<opset1::Interpolate>(layer);
    const auto attrs = interpolate->get_attrs();
    return attrs.mode == "nearest";
}

bool InterpolateTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    // TODO: expand transformation cases
    // just repeat CNNNetwork Resample transformation
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer);
    if (dequantization.empty()) {
        return false;
    }
    const auto interpolate = as_type_ptr<opset1::Interpolate>(layer);
    const auto interpAttrs = interpolate->get_attrs();

    if (interpAttrs.axes.count(0) || interpAttrs.axes.count(1)) {
        return false;
    }

    if (interpAttrs.mode != "nearest") {
        return false;
    }

    if (interpAttrs.pads_begin[0] != 0 || interpAttrs.pads_end[0] != 0 || interpAttrs.align_corners) {
        return false;
    }

    return true;
}
