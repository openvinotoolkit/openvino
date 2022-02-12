// Copyright (C) 2018-2022 Intel Corporation
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

/**
 * @ingroup ie_transformation_common_api
 * @brief WeightableLayerTransformation is base type for weightable operation transformation.
 */
class LP_TRANSFORMATIONS_API WeightableLayerTransformation : public LayerTransformation {
public:
    WeightableLayerTransformation(const Params& params);
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool canConvolutionBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer,
        const std::vector<ngraph::element::Type>& defaultPrecisions) const;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

    static bool checkPrecisionOnActivation(
        const std::shared_ptr<const ngraph::Node>& node,
        const std::vector<ngraph::element::Type>& supportedPrecisionsOnActivations) {
        return true;
    }

    static bool isQuantizedStatic(const std::shared_ptr<const Node>& layer,
        const bool reshapeIsRequired,
        const std::vector<ngraph::element::Type>& defaultPrecisions = precision_set::int8_support);

protected:
    bool decomposeFakeQuantizeForWeightsPath(const std::shared_ptr<Node>& weightableLayer, size_t outChannelsShapeIndex = 0ul) const;
    static bool isGroup(const std::shared_ptr<Node>& node);
    static bool isDepthwise(const std::shared_ptr<Node>& node);
    virtual size_t getInputChannels(const std::shared_ptr<ngraph::Node> conv) const = 0;

public:
    static std::shared_ptr<opset1::FakeQuantize> getFakeQuantizeOnWeights(const std::shared_ptr<Node>& node);
    static DataPrecision getDataPrecisionOnWeights(const std::shared_ptr<Node>& node, const std::vector<ngraph::element::Type>& defaultPrecisions);
    static bool isAsymmetricOnWeights(const std::shared_ptr<const Node>& node,
        const std::vector<ngraph::element::Type>& defaultPrecisions = precision_set::int8_support);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
