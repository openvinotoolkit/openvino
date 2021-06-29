// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>

namespace vpu {

namespace {

class CTCDecoderStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<CTCDecoderStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        auto cInd = input->desc().dimsOrder().dimInd(Dim::C);
        orderInfo.setOutput(outputEdge(0), output->desc().dimsOrder().createMovedDim(Dim::C, cInd));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::OnlyOne;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        input0->serializeBuffer(serializer);
        input1->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseCTCDecoder(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);
    const auto& ctcDecoder = ngraph::as_type_ptr<ngraph::op::v0::CTCGreedyDecoder>(node);
    IE_ASSERT(ctcDecoder != nullptr);
    auto ctc_merge_repeated_ = ctcDecoder->get_ctc_merge_repeated();
    if (ctc_merge_repeated_ != 1) {
        VPU_THROW_EXCEPTION
            << ctcDecoder->get_name() <<  " [" << ctcDecoder->get_type_name()
            << "] has incorrect ctc_merge_repeated param value."
            << " Kernel support case when ctc_merge_repeated_ == 1 only";
    }

    model->addNewStage<CTCDecoderStage>(ctcDecoder->get_name(), StageType::CTCDecoder, ctcDecoder, inputs, outputs);
}

}  // namespace vpu
