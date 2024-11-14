// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_sum.hpp"
#include <memory>

#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

ReduceSumTransformation::ReduceSumTransformation(const Params& params) : ReduceBaseTransformation(params) {
    MATCHER_SCOPE(ReduceSumTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::ReduceSum>({ pattern::wrap_type<ov::opset1::Multiply>(), pattern::wrap_type<ov::opset1::Constant>() });

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ReduceSumTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    const auto reduceSum = ov::as_type_ptr<ov::opset1::ReduceSum>(reduce);
    if (!reduceSum || !ReduceBaseTransformation::canBeTransformed(context, reduceSum)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(reduce, defaultPrecisions);
    if (dequantization.subtract) {
        const auto reductionAxes = reduceSum->get_reduction_axes();
        const auto inputPShape = dequantization.data.get_partial_shape();

        for (const auto& elem : reductionAxes) {
            if (inputPShape[elem].is_dynamic()) {
                return false;
            }
        }
    }

    return true;
}

void ReduceSumTransformation::changeDequantizationValues(
    const std::shared_ptr<Node>& reduce,
    FakeQuantizeDequantization& dequantization) const {
    ReduceBaseTransformation::changeDequantizationValues(reduce, dequantization);

    if (dequantization.subtract) {
        const auto reduceSum = ov::as_type_ptr<ov::opset1::ReduceSum>(reduce);
        const auto reductionAxes = reduceSum->get_reduction_axes();
        const auto inputShape = reduceSum->get_input_partial_shape(0);

        // calculating the number of reduced elements
        size_t reductionSize = 1ul;
        for (const auto& elem : reductionAxes) {
            reductionSize *= inputShape[elem].get_length();
        }

        // (a1 - s) + (a2 - s) + ... + (an - s) = (a1 + a2 + ... + an) - n * s
        const auto reductionSizeConstant = ov::opset1::Constant::create(deqPrecision, Shape{}, { static_cast<float>(reductionSize) });
        OPENVINO_ASSERT(deqPrecision == dequantization.subtract->get_input_element_type(0),
                        "dequantization precision ", deqPrecision,
                        " differs from zero point 0 input ", dequantization.subtract->get_input_element_type(0));
        const auto result = fold<ov::opset1::Multiply>(
            foldConvert(dequantization.subtractConstant, deqPrecision),
            reductionSizeConstant);

        replace_node(
            dequantization.subtractConvert != nullptr ?
                std::dynamic_pointer_cast<ov::Node>(dequantization.subtractConvert) :
                dequantization.subtractConstant,
            result);

        dequantization.subtractConstant = ov::as_type_ptr<ov::opset1::Constant>(result);
        dequantization.subtractConvert = nullptr;
    }
}

bool ReduceSumTransformation::isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept {
    return false;
}

bool ReduceSumTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
