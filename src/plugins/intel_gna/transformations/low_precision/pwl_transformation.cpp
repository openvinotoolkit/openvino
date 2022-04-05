// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_transformation.hpp"
#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <openvino/cc/ngraph/itt.hpp>
#include <ops/pwl.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

PWLTransformation::PWLTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(PWLTransformation);
    auto matcher = pattern::wrap_type<ov::intel_gna::op::Pwl>({
        pattern::wrap_type<opset1::Multiply>(),
        pattern::wrap_type<opset1::Constant>(),
        pattern::wrap_type<opset1::Constant>(),
        pattern::wrap_type<opset1::Constant>()
    });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "PWLTransformation");
    this->register_matcher(m, callback);
}

bool PWLTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    auto pwl = m.get_match_root();
    if (!canBeTransformed(context, pwl)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(pwl, defaultPrecisions);
    auto mul_const_f32 = foldConvert(dequantization.multiplyConstant, deqPrecision);
    auto new_m = ngraph::op::util::make_try_fold<opset1::Multiply>(
            mul_const_f32,
            foldConvert(pwl->get_input_node_shared_ptr(1), deqPrecision));
    auto new_b = foldConvert(pwl->get_input_node_shared_ptr(2), deqPrecision);
    auto new_knots = ngraph::op::util::make_try_fold<opset1::Divide>(
            foldConvert(pwl->get_input_node_shared_ptr(3), deqPrecision),
            mul_const_f32);
    if (dequantization.subtract) {
        auto sub_const_f32 = foldConvert(dequantization.subtractConstant, deqPrecision);
        new_b = ngraph::op::util::make_try_fold<opset1::Subtract>(
            new_b,
            ngraph::op::util::make_try_fold<opset1::Multiply>(new_m, sub_const_f32));
        new_knots = ngraph::op::util::make_try_fold<opset1::Add>(new_knots, sub_const_f32);
    }
    std::shared_ptr<Node> new_pwl = nullptr;
    if (!dequantization.data.get_element_type().is_real()) {
        new_pwl = std::make_shared<op::TypeRelaxed<ov::intel_gna::op::Pwl>>(
            std::vector<element::Type>{element::f32, element::f32, element::f32, element::f32},
            std::vector<element::Type>{ element::f32 },
            op::TemporaryReplaceOutputType(dequantization.data, element::f32).get(),
            op::TemporaryReplaceOutputType(new_m, element::f32).get(),
            op::TemporaryReplaceOutputType(new_b, element::f32).get(),
            op::TemporaryReplaceOutputType(new_knots, element::f32).get());
    } else {
        new_pwl = pwl->copy_with_new_inputs({
             dequantization.data,
             new_m,
             new_b,
             new_knots
        });
    }

    NetworkHelper::copyInfo(pwl, new_pwl);
    ov::replace_node(pwl, new_pwl);

    return true;
}

bool PWLTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    if (dequantization.multiply == nullptr) {
        return false;
    }
    if (dequantization.subtract && !NetworkHelper::isScalarLike(dequantization.subtractConstant)) {
        return false;
    }

    return NetworkHelper::isScalarLike(dequantization.multiplyConstant);
}

bool PWLTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
