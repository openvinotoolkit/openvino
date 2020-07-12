// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/avg_pool.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void AvgPoolTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::AvgPool>({ make_op_label<opset1::Multiply>() }));
}

void AvgPoolTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    const std::shared_ptr<Node> pooling = separateInStandaloneBranch(m.get_match_root());
    moveDequantizationAfter(context, pooling, NetworkHelper::getDequantization(pooling), false);
}

bool AvgPoolTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
