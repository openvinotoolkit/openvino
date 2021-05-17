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

class TRANSFORMATIONS_API WeightableLayerTransformation : public LayerTransformation{
public:
    WeightableLayerTransformation(const Params& params);
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool canConvolutionBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const;
    bool isQuantized(std::shared_ptr<Node> layer, bool reshapeIsRequired) const noexcept;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

protected:
    void decomposeFakeQuantizeForWeightsPath(const std::shared_ptr<Node>& weightableLayer, size_t outChannelsShapeIndex = 0ul) const;
    static bool isGroup(const std::shared_ptr<Node>& node);
    static bool isDepthwise(const std::shared_ptr<Node>& node);

    std::shared_ptr<opset1::FakeQuantize> getFakeQuantizeOnWeights(const std::shared_ptr<Node>& node) const;
    DataPrecision getDataPrecisionOnWeights(const std::shared_ptr<Node>& node) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
