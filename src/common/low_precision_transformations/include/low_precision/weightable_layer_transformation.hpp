// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "layer_transformation.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief WeightableLayerTransformation is base type for weightable operation transformation.
 */
class LP_TRANSFORMATIONS_API WeightableLayerTransformation : public LayerTransformation {
public:
    struct LP_TRANSFORMATIONS_API CanBeTransformedParams {
        CanBeTransformedParams(
            const bool constantWeight = true,
            const bool perTensorQuantizationOnData = true,
            const bool limitWeightsDataPrecision = true,
            const bool dynamicWeights = false) :
            constantWeight(constantWeight),
            perTensorQuantizationOnData(perTensorQuantizationOnData),
            limitWeightsDataPrecision(limitWeightsDataPrecision),
            dynamicWeights(dynamicWeights) {
        }

        // weights on constant path only
        const bool constantWeight;
        // data with per-tensor quantization only
        const bool perTensorQuantizationOnData;
        // limit weights by expected precisions
        const bool limitWeightsDataPrecision;
        const bool dynamicWeights;
    };

    WeightableLayerTransformation(const Params& params, const CanBeTransformedParams& canBeTransformedParams = {});

    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
    bool canConvolutionBeTransformed(const std::shared_ptr<Node>& layer,
                                     const ov::element::TypeVector& defaultPrecisions) const;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

    static bool isQuantizedStatic(const std::shared_ptr<const Node>& layer,
        const bool reshapeIsRequired,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support());

protected:
    std::tuple<bool, std::shared_ptr<Node>, std::shared_ptr<Node>> decomposeFakeQuantizeForWeightsPath(
            const std::shared_ptr<Node>& weightableLayer,
            size_t outChannelsShapeIndex = 0ul) const;
    static bool isGroup(const std::shared_ptr<Node>& node);
    static bool isDepthwise(const std::shared_ptr<Node>& node);
    virtual size_t getInputChannels(const std::shared_ptr<ov::Node> conv) const = 0;

public:
    static std::shared_ptr<ov::opset1::FakeQuantize> getFakeQuantizeOnWeights(const std::shared_ptr<Node>& node);
    static DataPrecision getDataPrecisionOnWeights(const std::shared_ptr<Node>& node, const std::vector<ov::element::Type>& defaultPrecisions);
    static bool isAsymmetricOnWeights(const std::shared_ptr<const Node>& node,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support());

private:
    const CanBeTransformedParams canBeTransformedParams;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
