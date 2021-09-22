// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformation_context.hpp"
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API WeightableLayerTransformation : public LayerTransformation{
public:
    WeightableLayerTransformation(const Params& params);
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool canConvolutionBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

    static bool checkPrecisionOnActivation(
        const std::shared_ptr<const ngraph::Node>& node,
        const std::vector<ngraph::element::Type>& supportedPrecisionsOnActivations) {
        return true;
    }

    static bool isQuantizedStatic(const std::shared_ptr<const Node>& layer, const bool reshapeIsRequired) noexcept;

protected:
    bool decomposeFakeQuantizeForWeightsPath(const std::shared_ptr<Node>& weightableLayer, size_t outChannelsShapeIndex = 0ul) const;
    static bool isGroup(const std::shared_ptr<Node>& node);
    static bool isDepthwise(const std::shared_ptr<Node>& node);

public:
    static std::shared_ptr<opset1::FakeQuantize> getFakeQuantizeOnWeights(const std::shared_ptr<Node>& node);
    static DataPrecision getDataPrecisionOnWeights(const std::shared_ptr<Node>& node);
    static bool isAsymmetricOnWeights(const std::shared_ptr<const Node>& node);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
