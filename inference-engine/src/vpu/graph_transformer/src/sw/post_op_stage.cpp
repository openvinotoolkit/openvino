// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/sw/post_op_stage.hpp>

#include <memory>

#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>

namespace vpu {

void PostOpStage::propagateDataOrderImpl() const {
    IE_ASSERT(!_inputEdges.empty());
    IE_ASSERT(_outputEdges.size() == 1);

    auto input = _inputEdges[0]->input();

    auto inDimsOrder = input->desc().dimsOrder();

    _orderInfo.setOutput(_outputEdges[0], inDimsOrder);
}

void PostOpStage::getDataStridesRequirementsImpl() const {
    IE_ASSERT(!_inputEdges.empty());
    IE_ASSERT(_outputEdges.size() == 1);

    auto input = _inputEdges[0]->input();

    StridesRequirement reqs;
    reqs.add(2, DimStride::Compact);

    _stridesInfo.setInput(_inputEdges[0], reqs);
    _stridesInfo.setOutput(_outputEdges[0], reqs);
}

void PostOpStage::finalizeDataLayoutImpl() {
}

void PostOpStage::getBatchSupportInfoImpl() const {
    IE_ASSERT(!_inputEdges.empty());
    IE_ASSERT(_outputEdges.size() == 1);
}

StageSHAVEsRequirements PostOpStage::getSHAVEsRequirementsImpl() const {
    // TODO: more SHAVEs leads to hang on public MTCNN network with U8 input
    return StageSHAVEsRequirements::TwoOrOne;
}

void PostOpStage::finalCheckImpl() const {
}

void PostOpStage::serializeDataImpl(BlobSerializer& serializer) const {
    IE_ASSERT(!_inputEdges.empty());
    IE_ASSERT(_outputEdges.size() == 1);
    IE_ASSERT(_tempBufferEdges.empty());

    auto input = _inputEdges[0]->input();
    auto output = _outputEdges[0]->output();

    input->serializeNewBuffer(serializer);
    output->serializeNewBuffer(serializer);

    for (int i = 1; i < _inputEdges.size(); ++i) {
        _inputEdges[i]->input()->serializeNewBuffer(serializer);
    }
}

}  // namespace vpu
