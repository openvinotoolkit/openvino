// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API ReshapeTransformation : public LayerTransformation {
public:
    ReshapeTransformation(const Params& params) : LayerTransformation(params) {}
    ~ReshapeTransformation() override {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;

    static bool canBeTransformed(
        const ngraph::Shape& subtractShape,
        const ngraph::Shape& multiplyShape,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& outputShape);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
