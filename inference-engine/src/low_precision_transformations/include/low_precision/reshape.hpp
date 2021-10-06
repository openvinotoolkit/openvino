// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API ReshapeTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapeTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;

    static bool canBeTransformed(
        const ngraph::Shape& subtractShape,
        const ngraph::Shape& multiplyShape,
        const ngraph::PartialShape& inputShape,
        const ngraph::PartialShape& outputShape);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
