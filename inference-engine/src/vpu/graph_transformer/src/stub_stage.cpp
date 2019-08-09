// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stub_stage.hpp>

#include <memory>
#include <vector>

#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>

namespace vpu {

StagePtr StubStage::cloneImpl() const {
    return std::make_shared<StubStage>(*this);
}

void StubStage::propagateScaleFactorsImpl(
        const SmallVector<float>& inputScales,
        ScalePropagationStep step) {
    if (_type == StageType::StubConv ||
        _type == StageType::StubFullyConnected ||
        _type == StageType::StubDeconv) {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto weights = _inputEdges[1]->input();
        auto biases = _inputEdges[2]->input();

        IE_ASSERT(weights->usage() == DataUsage::Const);
        IE_ASSERT(biases->usage() == DataUsage::Const || biases->usage() == DataUsage::Fake);

        auto inputScale = inputScales[0];

        _scaleInfo.setInput(_inputEdges[1], step == ScalePropagationStep::Propagate ? 1.0f : inputScale);
        if (biases->usage() == DataUsage::Const) {
            _scaleInfo.setInput(_inputEdges[2], inputScale);
        }
        _scaleInfo.setOutput(_outputEdges[0], inputScale);
    } else {
        IE_ASSERT(_type == StageType::StubMaxPool || _type == StageType::StubAvgPool);

        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        _scaleInfo.setOutput(_outputEdges[0], inputScales[0]);
    }
}

void StubStage::propagateDataOrderImpl() const {
    VPU_THROW_EXCEPTION << "Must be replaced with real stage";
}

void StubStage::getDataStridesRequirementsImpl() const {
    VPU_THROW_EXCEPTION << "Must be replaced with real stage";
}

void StubStage::finalizeDataLayoutImpl() {
    VPU_THROW_EXCEPTION << "Must be replaced with real stage";
}

void StubStage::getBatchSupportInfoImpl() const {
    if (_type == StageType::StubConv ||
        _type == StageType::StubFullyConnected ||
        _type == StageType::StubDeconv) {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto weights = _inputEdges[1]->input();
        auto biases = _inputEdges[2]->input();

        IE_ASSERT(weights->usage() == DataUsage::Const);
        IE_ASSERT(biases->usage() == DataUsage::Const || biases->usage() == DataUsage::Fake);

        _batchInfo.setInput(_inputEdges[0], BatchSupport::Split);
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    } else {
        IE_ASSERT(_type == StageType::StubMaxPool || _type == StageType::StubAvgPool);

        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        // Pooling will support batch by merging it with previous dimension.
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
