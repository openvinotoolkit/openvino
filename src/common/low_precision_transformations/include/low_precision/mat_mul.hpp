// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MatMulTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    MatMulTransformation(const Params& params = Params());
    bool transform(TransformationContext &context, ngraph::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    static bool is3DTensorOnActivations(const std::shared_ptr<const Node>& node);
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
