// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stub_stage.hpp>

#include <memory>

#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>

namespace vpu {

StagePtr StubStage::cloneImpl() const {
    return std::make_shared<StubStage>(*this);
}

DataMap<float> StubStage::propagateScaleFactorsImpl(
        const DataMap<float>& inputScales,
        ScalePropagationStep step) {
    DataMap<float> out;

    if (_type == StageType::StubConv ||
        _type == StageType::StubFullyConnected ||
        _type == StageType::StubDeconv) {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto weights = _inputEdges[1]->input();
        auto biases = _inputEdges[2]->input();
        auto output = _outputEdges[0]->output();

        IE_ASSERT(weights->usage() == DataUsage::Const);
        IE_ASSERT(biases->usage() == DataUsage::Const || biases->usage() == DataUsage::Fake);

        auto inputScale = inputScales.at(input);

        out[weights] = step == ScalePropagationStep::Propagate ? 1.0f : inputScale;
        if (biases->usage() == DataUsage::Const) {
            out[biases] = inputScale;
        }
        out[output] = inputScale;
    } else {
        IE_ASSERT(_type == StageType::StubMaxPool || _type == StageType::StubAvgPool);

        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        out[output] = inputScales.at(input);
    }

    return out;
}

DataMap<DimsOrder> StubStage::propagateDataOrderImpl() const {
    VPU_THROW_EXCEPTION << "Must be replaced with real stage";
}

DataMap<StridesRequirement> StubStage::getDataStridesRequirementsImpl() const {
    VPU_THROW_EXCEPTION << "Must be replaced with real stage";
}

void StubStage::finalizeDataLayoutImpl() {
    VPU_THROW_EXCEPTION << "Must be replaced with real stage";
}

DataMap<BatchSupport> StubStage::getBatchSupportInfoImpl() const {
    DataMap<BatchSupport> out;

    if (_type == StageType::StubConv ||
        _type == StageType::StubFullyConnected ||
        _type == StageType::StubDeconv) {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto weights = _inputEdges[1]->input();
        auto biases = _inputEdges[2]->input();
        auto output = _outputEdges[0]->output();

        IE_ASSERT(weights->usage() == DataUsage::Const);
        IE_ASSERT(biases->usage() == DataUsage::Const || biases->usage() == DataUsage::Fake);

        out[input] = BatchSupport::Split;
        out[output] = BatchSupport::Split;
    } else {
        IE_ASSERT(_type == StageType::StubMaxPool || _type == StageType::StubAvgPool);

        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        // Pooling will support batch by merging it with previous dimension.
    }

    return out;
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
