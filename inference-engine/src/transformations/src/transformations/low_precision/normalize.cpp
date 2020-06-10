// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/normalize.hpp"

#include <algorithm>
#include <string>
#include <memory>
#include <vector>

#include "ngraph_ops/multiply_add.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

bool NormalizeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    return true;
}

void NormalizeTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::NormalizeL2>({ make_op_label<ngraph::op::MultiplyAdd>() }));
}

void NormalizeTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
}

//bool NormalizeTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
//    return false;
//}
