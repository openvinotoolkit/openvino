// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/max_pool.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

MaxPoolTransformation::MaxPoolTransformation(const Params& params) : LayerTransformation(params) {
}

void MaxPoolTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MaxPool>({ make_op_label<opset1::Multiply>() }));
}

void MaxPoolTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    if (!LayerTransformation::canBeTransformed(context, m.get_match_root())) {
        return;
    }

    const std::shared_ptr<Node> pooling = separateInStandaloneBranch(m.get_match_root());
    moveDequantizationAfter(context, pooling, NetworkHelper::getDequantization(pooling), true);
}

bool MaxPoolTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
