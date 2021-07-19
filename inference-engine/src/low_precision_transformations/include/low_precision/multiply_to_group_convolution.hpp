// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"
#include "common/operation_precision_restriction.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MultiplyToGroupConvolutionTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyToGroupConvolutionTransformation(
        const Params& params = Params(),
        const OperationPrecisionRestriction::PrecisionsByPort& restrictions = {});
    ~MultiplyToGroupConvolutionTransformation() override {}
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool isQuantized(const std::shared_ptr<const Node>& layer) const noexcept override;
    static bool canBeTransformedToGroupConvolution(const std::shared_ptr<const Node>& layer) noexcept;

    void setGroupSize(const size_t groupSize);
    size_t getGroupSize() const;
private:
    OperationPrecisionRestriction::PrecisionsByPort restrictions;
    size_t groupSize;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
