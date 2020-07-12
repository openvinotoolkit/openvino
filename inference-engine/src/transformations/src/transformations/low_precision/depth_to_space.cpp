// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/depth_to_space.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

void DepthToSpaceTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    TransparentBaseTransformation::transform(context, m);
}

void DepthToSpaceTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::DepthToSpace>({ make_op_label<ngraph::op::DepthToSpace>() }));
}

//bool DepthToSpaceTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
//    return true;
//}

bool DepthToSpaceTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!TransparentBaseTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    return true;
}
