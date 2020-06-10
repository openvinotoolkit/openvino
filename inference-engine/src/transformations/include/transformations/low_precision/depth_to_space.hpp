// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transparent_base_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API DepthToSpaceTransformation : public TransparentBaseTransformation {
public:
    DepthToSpaceTransformation(const Params& params) : TransparentBaseTransformation(params) {}
    ~DepthToSpaceTransformation() override {}
    void transform(TransformationContext &context, ngraph::pattern::Matcher &m) const override;
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    //bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
