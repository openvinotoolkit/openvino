// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API NormalizeTransformation : public LayerTransformation {
public:
    NormalizeTransformation(const Params& params) : LayerTransformation(params) {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext &context, ngraph::pattern::Matcher &m) const override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    //bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
