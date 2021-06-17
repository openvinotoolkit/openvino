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

// void FrontEnd::parseBatchNorm(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
//     IE_ASSERT(inputs.size() == 1);
//     IE_ASSERT(outputs.size() == 1);

//     const auto& batchNorm = ngraph::as_type_ptr<ngraph::opset4::BatchNormInference>(node);
//     VPU_THROW_UNLESS(batchNorm != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", batchNorm->get_friendly_name(), batchNorm->get_type_name());

//     auto input = inputs[0];
//     auto output = outputs[0];

//     Data origWeights, origBiases;
    
//     std::tie(origWeights, origBiases) = getWeightsAndBiases(model, batchNorm);

//     IE_ASSERT(origWeights->desc().totalDimSize() >= input->desc().dim(Dim::C));
//     auto weights = model->duplicateData(origWeights, "@batch-norm", DataDesc({input->desc().dim(Dim::C)}),
//         std::make_shared<BatchNormalizationWeightsContent>(origWeights->content(), batchNorm->get_eps_value()));

//     if (origBiases->usage() != DataUsage::Fake) {
//         IE_ASSERT(origBiases->desc().totalDimSize() >= input->desc().dim(Dim::C));
//         auto biases = model->duplicateData(origBiases, "@batch-norm", DataDesc({input->desc().dim(Dim::C)}),
//             std::make_shared<BatchNormalizationBiasesContent>(origBiases->content(), weights->content()));

//         auto tempOutput = model->duplicateData(output, "@temp");

//         _stageBuilder->addBiasStage(model, batchNorm->get_friendly_name(), batchNorm, tempOutput, biases, output);

//         output = tempOutput;
//     }

//     _stageBuilder->addScaleStage(model, batchNorm->get_friendly_name(), batchNorm, input, weights, output);
// }

}  // namespace vpu
