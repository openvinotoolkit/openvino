// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/model/data_contents/batch_norm_contents.hpp>

#include <precision_utils.h>

#include <cmath>
#include <vector>
#include <memory>

namespace vpu {

void FrontEnd::parseBatchNorm(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::BatchNormalizationLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];
    auto output = outputs[0];

    Data origWeights, origBiases;
    std::tie(origWeights, origBiases) = getWeightsAndBiases(model, layer);

    IE_ASSERT(origWeights->desc().totalDimSize() >= input->desc().dim(Dim::C));
    auto weights = model->duplicateData(origWeights, "@batch-norm", DataDesc({input->desc().dim(Dim::C)}),
        std::make_shared<BatchNormalizationWeightsContent>(origWeights->content(), layer->epsilon));

    if (origBiases->usage() != DataUsage::Fake) {
        IE_ASSERT(origBiases->desc().totalDimSize() >= input->desc().dim(Dim::C));
        auto biases = model->duplicateData(origBiases, "@batch-norm", DataDesc({input->desc().dim(Dim::C)}),
            std::make_shared<BatchNormalizationBiasesContent>(origBiases->content(), weights->content()));

        auto tempOutput = model->duplicateData(output, "@temp");

        _stageBuilder->addBiasStage(model, layer->name, layer, tempOutput, biases, output);

        output = tempOutput;
    }

    _stageBuilder->addScaleStage(model, layer->name, layer, input, weights, output);
}

}  // namespace vpu
