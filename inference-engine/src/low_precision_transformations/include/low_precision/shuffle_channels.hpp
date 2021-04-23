// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API ShuffleChannelsTransformation : public LayerTransformation {
public:
    ShuffleChannelsTransformation(const Params& params);
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    bool transform(TransformationContext& context, ngraph::pattern::Matcher& m) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
