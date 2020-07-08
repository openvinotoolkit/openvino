// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "transformation_context.hpp"
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class PrecisionsInfo {
public:
    PrecisionsInfo(const element::Type original, const element::Type low) : original(original), low(low) {}
    const element::Type original;
    const element::Type low;
};


class TRANSFORMATIONS_API WeightableLayerTransformation : public LayerTransformation{
public:
    WeightableLayerTransformation(const Params& params);
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;

    bool isQuantized(std::shared_ptr<Node> layer) const noexcept;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept;

#if 0 // TODO: LPT-TO-NGRAPH
protected:
    void updateLayerBiases(
        TransformationContext& context,
        const CNNLayer& convolution,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts,
        std::vector<float>& biasesShifts) const;

    void updateWeights(
        const CNNLayerPtr fakeQuantize,
        std::vector<float>& outputLowValues,
        std::vector<float>& outputHighValues) const;

    void updateToSupportAsymmetricQuantization(
        TransformationContext& context,
        const CNNLayer& layer,
        const PrecisionsInfo& dataPrecisionsInfo,
        std::vector<float>& dataShifts,
        const PrecisionsInfo& weightsPrecisionsInfo,
        std::vector<float>& weightsShifts) const;

    void createAsymmetric(
        TransformationContext& context,
        const CNNLayer& parent,
        const CNNLayer& child,
        const PrecisionsInfo& precisionsInfo,
        const std::vector<float>& quantizationShifts,
        const bool onWeights) const;

#endif

    DataPrecision decomposeFakeQuantizeForWeightsPath(
            std::shared_ptr<Node> weightableLayer,
            const bool supportAsymmetricQuantization) const;

    static bool isDepthwise(std::shared_ptr<Node> layer);
};

typedef std::shared_ptr<WeightableLayerTransformation> WeightableLayerTransformationPtr;

} // namespace low_precision
} // namespace pass
} // namespace ngraph
