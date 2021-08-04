// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API EltwiseBaseTransformation : public LayerTransformation {
public:
    EltwiseBaseTransformation(const Params& params) : LayerTransformation(params) {}
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

    static bool isBroadcasted(const PartialShape& shape) noexcept;
protected:
    int getNotEmpty(const std::shared_ptr<Node>& eltwise) const;
    // Return indexes:
    // 1. first  - data branch index for eltwise
    // 2. second - Constant branch index for data branch Multiply
    std::pair<int, int> getMultiplyConstBranch(const std::shared_ptr<Node>& eltwise) const;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
