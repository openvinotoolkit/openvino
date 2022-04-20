// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API AssignAndReadValueTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("AssignAndReadValueTransformation", "0");
    AssignAndReadValueTransformation(const std::shared_ptr<ngraph::Function> function, const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher& m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
private:
    std::shared_ptr<ngraph::Function> function;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
