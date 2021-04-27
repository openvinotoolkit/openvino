// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/avg_pool.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

AvgPoolTransformation::AvgPoolTransformation(const Params& params) : LayerTransformation(params) {
}

void AvgPoolTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::AvgPool>({ make_op_label<opset1::Multiply>() }));
}

bool AvgPoolTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> pooling = NetworkHelper::separateInStandaloneBranch(m.get_match_root());

    const std::vector<std::shared_ptr<ngraph::Node>> children = getChildrenRecursivelyExceptPrecisionPreserved(pooling);

    bool updatePrecision;
    if ((children.size() == 1ul) && (!this->layerTransformationsManager->isQuantized(children[0]))) {
        updatePrecision = false;
    } else {
        updatePrecision = NetworkHelper::notAllChildrensAreFQ(children);
    }

    moveDequantizationAfter(context, pooling, NetworkHelper::getDequantization(pooling), updatePrecision);
    return true;
}

bool AvgPoolTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    auto dequantization = NetworkHelper::getDequantization(operation);

    return !!dequantization.multiply;
}

bool AvgPoolTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    const std::vector<std::shared_ptr<ngraph::Node>> children = getChildrenRecursivelyExceptPrecisionPreserved(layer);
    return NetworkHelper::notAllChildrensAreFQ(children);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
