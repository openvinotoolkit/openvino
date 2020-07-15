// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API EltwiseBaseTransformation : public LayerTransformation {
public:
    EltwiseBaseTransformation(const Params& params) : LayerTransformation(params) {}
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;

protected:
    int getNotEmpty(const std::shared_ptr<Node>& eltwise) const;
    std::pair<int, int> getMultiplyConstBranch(const std::shared_ptr<Node>& eltwise) const;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
