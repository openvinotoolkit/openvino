// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/layer_transformation.hpp"
#include "transformations/low_precision/eltwise_base_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API SubtractMultiplyToMultiplyAddTransformation : public LayerTransformation {
public:
    SubtractMultiplyToMultiplyAddTransformation(const Params& params) : LayerTransformation(params) {}
    ~SubtractMultiplyToMultiplyAddTransformation() override {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
