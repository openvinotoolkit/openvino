// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mat_mul.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ngraph_ops/multiply_add.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

void MatMulTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
}

void MatMulTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<ngraph::op::MultiplyAdd>(), make_op_label<ngraph::op::MultiplyAdd>() }));
}

//bool MatMulTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
//    return true;
//}

bool MatMulTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    return true;
}
