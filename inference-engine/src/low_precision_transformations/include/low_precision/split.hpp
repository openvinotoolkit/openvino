// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "layer_transformation.hpp"
#include "ngraph/node.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API SplitTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    SplitTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pattern::Matcher& m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    void updateOutputs(
        TransformationContext& context,
        std::vector<std::shared_ptr<ov::Node>> lastNodes,
        std::shared_ptr<ov::Node> originalNode) const;
};
} // namespace low_precision
} // namespace pass
} // namespace ov
