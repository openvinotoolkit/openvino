// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/sw/post_op_stage.hpp>

#include <memory>

#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>

namespace vpu {

void PostOpStage::propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) {
    auto input = inputEdge(0)->input();

    auto inDimsOrder = input->desc().dimsOrder();

    orderInfo.setOutput(outputEdge(0), inDimsOrder);
}

void PostOpStage::getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) {
    auto input = inputEdge(0)->input();

    StridesRequirement reqs;
    reqs.add(2, DimStride::Compact);

    stridesInfo.setInput(inputEdge(0), reqs);
    stridesInfo.setOutput(outputEdge(0), reqs);
}

void PostOpStage::finalizeDataLayoutImpl() {
}

void PostOpStage::getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) {
}

StageSHAVEsRequirements PostOpStage::getSHAVEsRequirementsImpl() const {
    // TODO: more SHAVEs leads to hang on public MTCNN network with U8 input
    return StageSHAVEsRequirements::TwoOrOne;
}

void PostOpStage::initialCheckImpl() const {
    IE_ASSERT(numInputs() > 0);
    IE_ASSERT(numOutputs() == 1);
    assertAllInputsOutputsTypes(this, DataType::FP16, DataType::FP16);
}

void PostOpStage::serializeDataImpl(BlobSerializer& serializer) const {
    auto input = inputEdge(0)->input();
    auto output = outputEdge(0)->output();

    input->serializeNewBuffer(serializer);
    output->serializeNewBuffer(serializer);

    for (int i = 1; i < numInputs(); ++i) {
        this->input(i)->serializeNewBuffer(serializer);
    }
}

}  // namespace vpu
