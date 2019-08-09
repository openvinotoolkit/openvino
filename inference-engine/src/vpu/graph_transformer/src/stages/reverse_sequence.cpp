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

#include <vector>
#include <memory>
#include <set>

namespace vpu {

namespace {

class ReverseSequenceStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ReverseSequenceStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        _orderInfo.setOutput(_outputEdges[0], input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl() const override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto seq_lengths = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        auto seq_axis = input->desc().dimsOrder().dimInd(attrs().get<Dim>("seq_axis"));
        auto batch_axis = input->desc().dimsOrder().dimInd(attrs().get<Dim>("batch_axis"));

        serializer.append(static_cast<int32_t>(seq_axis));
        serializer.append(static_cast<int32_t>(batch_axis));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto seq_lengths = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        input->serializeNewBuffer(serializer);
        seq_lengths->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseReverseSequence(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<ReverseSequenceStage>(
        layer->name,
        StageType::ReverseSequence,
        layer,
        inputs,
        outputs);

    auto input = inputs[0];

    auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    auto seq_axis = layer->GetParamAsInt("seq_axis");
    auto seq_axis_index = perm[input->desc().numDims() - 1 - seq_axis];
    auto batch_axis = layer->GetParamAsInt("batch_axis");
    auto batch_axis_index = perm[input->desc().numDims() - 1 - batch_axis];

    stage->attrs().set<Dim>("seq_axis", seq_axis_index);
    stage->attrs().set<Dim>("batch_axis", batch_axis_index);
}

}  // namespace vpu
