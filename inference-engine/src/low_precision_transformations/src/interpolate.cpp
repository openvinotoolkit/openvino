// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/interpolate.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::InterpolateTransformation, "InterpolateTransformation", 0);

InterpolateTransformation::InterpolateTransformation(const Params& params) : LayerTransformation(params) {
    auto mul = pattern::wrap_type<opset1::Multiply>();

    auto interpolate1 = pattern::wrap_type<opset1::Interpolate>({
        mul,
        pattern::wrap_type<opset1::Constant>() });

    auto interpolate4 = pattern::wrap_type<opset4::Interpolate>({
        mul,
        pattern::wrap_type<opset1::Constant>(),
        pattern::wrap_type<opset1::Constant>() });

    auto interpolate4_2 = pattern::wrap_type<opset4::Interpolate>({
        mul,
        pattern::wrap_type<opset1::Constant>(),
        pattern::wrap_type<opset1::Constant>(),
        pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(
        std::make_shared<pattern::op::Or>(OutputVector{ interpolate1, interpolate4, interpolate4_2 }),
        "InterpolateTransformation");

    this->register_matcher(matcher, callback);
}

bool InterpolateTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<Node> interpolate = m.get_match_root();
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }
    interpolate = NetworkHelper::separateInStandaloneBranch(interpolate);
    moveDequantizationAfter(context, interpolate, NetworkHelper::getDequantization(interpolate), true);
    return true;
}

bool InterpolateTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    std::shared_ptr<opset1::Interpolate> interpolate1 = as_type_ptr<opset1::Interpolate>(layer);
    if (interpolate1) {
        const auto attrs = interpolate1->get_attrs();
        return attrs.mode == "nearest";
    }

    std::shared_ptr<opset4::Interpolate> interpolate4 = as_type_ptr<opset4::Interpolate>(layer);
    if (interpolate4) {
        const auto attrs = interpolate4->get_attrs();
        return attrs.mode == op::v4::Interpolate::InterpolateMode::nearest;
    }

    return false;
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

    const auto interpolate1 = as_type_ptr<opset1::Interpolate>(layer);
    if (interpolate1) {
        const auto interpAttrs = interpolate1->get_attrs();
        if (interpAttrs.axes.count(0) || interpAttrs.axes.count(1)) {
            return false;
        }
        if (interpAttrs.mode != "nearest") {
            return false;
        }
        if (interpAttrs.pads_begin[0] != 0 || interpAttrs.pads_end[0] != 0 || interpAttrs.align_corners) {
            return false;
        }
    }

    const auto interpolate4 = as_type_ptr<opset4::Interpolate>(layer);
    if (interpolate4) {
        const auto interpAttrs = interpolate4->get_attrs();

        if (interpAttrs.mode != op::v4::Interpolate::InterpolateMode::nearest) {
            return false;
        }

        auto pads_begin = interpAttrs.pads_begin;
        for (size_t i = 0; i < pads_begin.size(); ++i) {
            if (pads_begin[i] != 0) {
                return false;
            }
        }

        auto pads_end = interpAttrs.pads_end;
        for (size_t i = 0; i < pads_end.size(); ++i) {
            if (pads_end[i] != 0) {
                return false;
            }
        }

        if (interpAttrs.coordinate_transformation_mode == op::v4::Interpolate::CoordinateTransformMode::align_corners) {
            return false;
        }
    }

    return true;
}
