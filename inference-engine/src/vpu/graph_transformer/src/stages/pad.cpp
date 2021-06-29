// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>
#include <vector>
#include <list>
#include <set>
#include <unordered_set>
#include <memory>

namespace vpu {

namespace {

class PadStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PadStage>(*this);
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
        // TODO: try merge with last dimension
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();

        auto perm = input->desc().dimsOrder().toPermutation();
        IE_ASSERT(perm.size() <= 4);

        auto pad_value = attrs().get<float>("pad_value");
        auto pad_mode = attrs().get<PadMode>("pad_mode");
        const auto& pads_begin = attrs().get<DimValues>("pads_begin");
        const auto& pads_end = attrs().get<DimValues>("pads_end");

        int i = 0;
        for (; i < perm.size(); ++i) {
            serializer.append(static_cast<uint32_t>(pads_begin.get(perm[i], 0)));
            serializer.append(static_cast<uint32_t>(pads_end.get(perm[i], 0)));
        }
        for (; i < 4; ++i) {
            serializer.append(static_cast<uint32_t>(0));
            serializer.append(static_cast<uint32_t>(0));
        }

        serializer.append(static_cast<float>(pad_value));
        serializer.append(static_cast<uint32_t>(pad_mode));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace
float getPadValue (const NodePtr& pad) {
    auto result = 0.f;
    if (pad->inputs().size() == 4) {
        auto const_node =
            std::dynamic_pointer_cast<ngraph::opset4::Constant>(pad->input(3).get_source_output().get_node_shared_ptr());
        if (!const_node) {
            VPU_THROW_UNLESS(false, "Pad {} with not constant pad_value is not allowed", pad->get_friendly_name());
        }
        //rework
        result = const_node->get_vector<float>()[0];
    }
    return result;
}
void FrontEnd::parsePad(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    const auto& pad = ngraph::as_type_ptr<ngraph::opset4::Pad>(node);
    IE_ASSERT(pad != nullptr);
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    const auto ndims = inputs[0]->desc().dimsOrder().numDims();
    VPU_THROW_UNLESS(ndims == 3 || ndims == 4, "Layer %s support only 3D and 4D input, but %dD provided", pad->get_name(), ndims);

    VPU_THROW_UNLESS(pad->get_pads_begin().size() <= 4, "Layer %s support pads_begin size less than or equal 4, but %d provided",
                     pad->get_friendly_name(), pad->get_pads_begin().size());
    VPU_THROW_UNLESS(pad->get_pads_end().size() <= 4, "Layer %s support pads_end size less than or equal 4, but %d provided",
                     pad->get_friendly_name(), pad->get_pads_end().size());

    DimsOrder dimsOrder = inputs[0]->desc().dimsOrder();
    auto padsBegin = pad->get_pads_begin();
    auto padsEnd = pad->get_pads_end();
    DimValues pads_begin;
    pads_begin.set(Dim::W, dimsOrder.hasDim(Dim::W) ? padsBegin[dimToIeInd(Dim::W, ndims)] : 0);
    pads_begin.set(Dim::H, dimsOrder.hasDim(Dim::H) ? padsBegin[dimToIeInd(Dim::H, ndims)] : 0);
    pads_begin.set(Dim::C, dimsOrder.hasDim(Dim::C) ? padsBegin[dimToIeInd(Dim::C, ndims)] : 0);
    pads_begin.set(Dim::N, dimsOrder.hasDim(Dim::N) ? padsBegin[dimToIeInd(Dim::N, ndims)] : 0);

    DimValues pads_end;
    pads_end.set(Dim::W, dimsOrder.hasDim(Dim::W) ? padsEnd[dimToIeInd(Dim::W, ndims)] : 0);
    pads_end.set(Dim::H, dimsOrder.hasDim(Dim::H) ? padsEnd[dimToIeInd(Dim::H, ndims)] : 0);
    pads_end.set(Dim::C, dimsOrder.hasDim(Dim::C) ? padsEnd[dimToIeInd(Dim::C, ndims)] : 0);
    pads_end.set(Dim::N, dimsOrder.hasDim(Dim::N) ? padsEnd[dimToIeInd(Dim::N, ndims)] : 0);
    auto padValue = getPadValue(pad);

    _stageBuilder->addPadStage(
        model,
        pad->get_friendly_name(),
        pad,
        static_cast<PadMode>(pad->get_pad_mode()),
        padValue,
        pads_begin,
        pads_end,
        inputs[0],
        outputs[0]);
}

Stage StageBuilder::addPadStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        PadMode padMode,
        float pad_value,
        const DimValues& pads_begin,
        const DimValues& pads_end,
        const Data& input,
        const Data& output) {
    auto stage = model->addNewStage<PadStage>(
        name,
        StageType::Pad,
        node,
        {input},
        {output});

    stage->attrs().set<float>("pad_value", pad_value);
    stage->attrs().set<PadMode>("pad_mode", padMode);
    stage->attrs().set<DimValues>("pads_begin", pads_begin);
    stage->attrs().set<DimValues>("pads_end", pads_end);

    return stage;
}

}  // namespace vpu
