// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_sum.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::ReduceSumTransformation, "ReduceSumTransformation", 0);

ReduceSumTransformation::ReduceSumTransformation(const Params& params) : ReduceBaseTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::ReduceSum>({ pattern::wrap_type<opset1::Multiply>(), pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "ReduceSumTransformation");
    this->register_matcher(m, callback);
}

bool ReduceSumTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    const auto reduceSum = as_type_ptr<opset1::ReduceSum>(reduce);
    if (!reduceSum || !ReduceBaseTransformation::canBeTransformed(context, reduceSum)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(reduce);
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
        const auto reduceSum = as_type_ptr<opset1::ReduceSum>(reduce);
        const auto reductionAxes = reduceSum->get_reduction_axes();
        const auto inputShape = reduceSum->get_input_partial_shape(0);

        // calculating the number of reduced elements
        size_t reductionSize = 1ul;
        for (const auto& elem : reductionAxes) {
            reductionSize *= inputShape[elem].get_length();
        }

        // (a1 - s) + (a2 - s) + ... + (an - s) = (a1 + a2 + ... + an) - n * s
        const auto reductionSizeConstant = opset1::Constant::create(deqPrecision, Shape{}, { static_cast<float>(reductionSize) });
        const auto result = fold<opset1::Multiply>(dequantization.subtractConstant, reductionSizeConstant);

        replace_node(dequantization.subtractConstant, result);
        dequantization.subtractConstant = as_type_ptr<opset1::Constant>(result);
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
} // namespace ngraph
