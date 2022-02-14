// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API LSTMTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("LSTMTransformation", "0");
    LSTMTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    void propagateSkipCleanupAttribute(std::shared_ptr<Node> dequantization_multiply);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
