//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <vpu/frontend/frontend.hpp>

#include <string>
#include <vector>
#include <list>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

void FrontEnd::parseGather(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);
    auto layer = std::dynamic_pointer_cast<ie::GatherLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];

    IE_ASSERT(layer->axis < input->desc().numDims());

    auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    auto axis = perm[input->desc().numDims() - 1 - layer->axis];

    _stageBuilder->addGatherStage(model, layer->name, layer, inputs[0], inputs[1], outputs[0], axis);
}

namespace {

class GatherStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<GatherStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        if (step == ScalePropagationStep::Propagate) {
            _scaleInfo.setOutput(_outputEdges[0], inputScales[0]);
        } else {
            // Gather can only propagate scaling.
            for (const auto& inEdge : _inputEdges) {
                _scaleInfo.setInput(inEdge, 1.0f);
            }
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        }
    }

    void propagateDataOrderImpl() const override {
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        for (const auto& inEdge : _inputEdges) {
            _stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        _stridesInfo.setOutput(_outputEdges[0], StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 2);

        auto input = _inputEdges[0]->input();

        auto axis = attrs().get<Dim>("axis");
        auto axisInd = input->desc().dimsOrder().dimInd(axis);

        serializer.append(static_cast<int>(axisInd));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        input0->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
        input1->serializeNewBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addGatherStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input0,
        const Data& input1,
        const Data& output,
        Dim axis) {
    auto stage = model->addNewStage<GatherStage>(
        name,
        StageType::Gather,
        layer,
        {input0, input1},
        {output});

    stage->attrs().set<Dim>("axis", axis);

    return stage;
}

}  // namespace vpu
