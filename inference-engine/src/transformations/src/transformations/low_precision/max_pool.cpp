// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/max_pool.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_ops/multiply_add.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void MaxPoolTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MaxPool>({ make_op_label<opset1::Multiply>() }));
}

void MaxPoolTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    // std::shared_ptr<Node> pooling = separateInStandaloneBranch(m.get_match_root());
    std::shared_ptr<Node> pooling = m.get_match_root();
    const auto result = insertDequantization(pooling, getDequantization(pooling), true);
    updateOutput(context, result.lastDequantization, result.newOperation);
}

bool MaxPoolTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
