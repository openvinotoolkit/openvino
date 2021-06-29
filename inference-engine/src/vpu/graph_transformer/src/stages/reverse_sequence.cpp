// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>

namespace vpu {

namespace {

class ReverseSequenceStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ReverseSequenceStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
         auto input = inputEdge(0)->input();
         orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
         auto input = inputEdge(0)->input();
         auto seq_lengths = inputEdge(1)->input();
         auto output = outputEdge(0)->output();

         auto seq_axis = input->desc().dimsOrder().dimInd(attrs().get<Dim>("seq_axis"));
         auto batch_axis = input->desc().dimsOrder().dimInd(attrs().get<Dim>("batch_axis"));

         serializer.append(static_cast<int32_t>(seq_axis));
         serializer.append(static_cast<int32_t>(batch_axis));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
         auto input = inputEdge(0)->input();
         auto seq_lengths = inputEdge(1)->input();
         auto output = outputEdge(0)->output();

         input->serializeBuffer(serializer);
         seq_lengths->serializeBuffer(serializer);
         output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseReverseSequence(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    const auto& reverseSequence = ngraph::as_type_ptr<ngraph::op::v0::ReverseSequence>(node);
    IE_ASSERT(reverseSequence != nullptr);
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<ReverseSequenceStage>(reverseSequence->get_name(), StageType::ReverseSequence, reverseSequence, inputs, outputs);

    auto input = inputs[0];

    auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    auto seq_axis = reverseSequence->get_sequence_axis();
    auto seq_axis_index = perm[input->desc().numDims() - 1 - seq_axis];
    auto batch_axis = reverseSequence->get_batch_axis();
    auto batch_axis_index = perm[input->desc().numDims() - 1 - batch_axis];

    stage->attrs().set<Dim>("seq_axis", seq_axis_index);
    stage->attrs().set<Dim>("batch_axis", batch_axis_index);
}

}  // namespace vpu
