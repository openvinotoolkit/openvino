// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stages/stub_stage.hpp>

#include <memory>
#include <vector>
#include <utility>

#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>

namespace vpu {

StagePtr StubStage::cloneImpl() const {
    return std::make_shared<StubStage>(*this);
}

void StubStage::propagateDataOrderImpl(StageDataInfo<DimsOrder>&) {
    VPU_THROW_EXCEPTION << "Must be replaced with real stage";
}

void StubStage::getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>&) {
    VPU_THROW_EXCEPTION << "Must be replaced with real stage";
}

void StubStage::finalizeDataLayoutImpl() {
    VPU_THROW_EXCEPTION << "Must be replaced with real stage";
}

void StubStage::getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) {
    if (type() == StageType::StubConv ||
        type() == StageType::StubFullyConnected ||
        type() == StageType::StubDeconv) {
        auto weights = inputEdge(1)->input();
        auto biases = inputEdge(2)->input();

        IE_ASSERT(weights->usage() == DataUsage::Const);
        IE_ASSERT(biases->usage() == DataUsage::Const || biases->usage() == DataUsage::Fake);

        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    } else {
        IE_ASSERT(type() == StageType::StubMaxPool || type() == StageType::StubAvgPool);

        // Pooling will support batch by merging it with previous dimension.
    }
}

void StubStage::initialCheckImpl() const {
    if (type() == StageType::StubConv || type() == StageType::StubFullyConnected || type() == StageType::StubDeconv) {
        assertInputsOutputsTypes(this,
            {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}, {DataType::FP16}},
            {{DataType::FP16}});
    } else if (type() == StageType::StubMaxPool || type() == StageType::StubAvgPool) {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    } else {
        VPU_THROW_EXCEPTION << "unknown type";
    }
}

void StubStage::finalCheckImpl() const {
    VPU_THROW_EXCEPTION << "Must never be called";
}

void StubStage::serializeParamsImpl(BlobSerializer&) const {
    VPU_THROW_EXCEPTION << "Must never be called";
}

void StubStage::serializeDataImpl(BlobSerializer&) const {
    VPU_THROW_EXCEPTION << "Must never be called";
}

}  // namespace vpu
