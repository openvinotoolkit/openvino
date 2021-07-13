// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <ngraph/ngraph.hpp>

#include "layer_transformation.hpp"
#include "common/fake_quantize_dequantization.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API ConcatTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    ConcatTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;

protected:
    static bool isHandled(
        const TransformationContext& context,
        const std::vector<std::shared_ptr<ngraph::Node>>& quantizationOperations);

    void fillDequantizationNodes(
        const std::vector<FakeQuantizeDequantization>& layerDequantizations,
        const std::shared_ptr<Node> layer,
        NodeVector& convertNodes,
        NodeVector& subtractNodes,
        NodeVector& multiplyNodes) const;

    std::shared_ptr<Node> concatenateDeqNodes(NodeVector& nodes) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
