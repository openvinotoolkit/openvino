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

class TRANSFORMATIONS_API TransparentBaseTransformation : public LayerTransformation {
public:
    TransparentBaseTransformation(const Params& params) : LayerTransformation(params) {}
    ~TransparentBaseTransformation() override {};
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
