// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <vector>
#include <map>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class NormalizeStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<NormalizeStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        if (input(0)->desc().dimsOrder().dimInd(Dim::C) == 0) {
            stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
            stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto acrossSpatial = attrs().get<bool>("acrossSpatial");
        auto channelShared = attrs().get<bool>("channelShared");
        auto eps = attrs().get<float>("eps");

        serializer.append(static_cast<int32_t>(acrossSpatial));
        serializer.append(static_cast<int32_t>(channelShared));
        serializer.append(static_cast<float>(eps));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto scales = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        scales->serializeBuffer(serializer);
    }
    struct NormalizeParams {
        int acrossSpatial;
        int channelShared;
    };

    NormalizeParams getParamsFromNode(const NodePtr& node) {
        auto normalize = ngraph::as_type_ptr<ngraph::opset4::NormalizeL2>(node);
        IE_ASSERT(normalize != nullptr);
        auto const_axis = std::dynamic_pointer_cast<ngraph::opset4::Constant> (normalize->input(1).get_source_output().get_node_shared_ptr());
        IE_ASSERT(const_axis != nullptr);

        NormalizeParams outputParams;
        auto axis = const_axis->cast_vector<size_t>();
        outputParams.acrossSpatial = !(axis.size() == 1 && axis[0] == 1);
        outputParams.channelShared = true;
        return outputParams;
    }
};

}  // namespace


void FrontEnd::parseNormalize(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    const auto& normalize = ngraph::as_type_ptr<ngraph::opset4::NormalizeL2>(node);
    VPU_THROW_UNLESS(normalize != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    int acrossSpatial;
    int channelShared;
    auto const_axis = std::dynamic_pointer_cast<ngraph::opset4::Constant>(normalize->input(1).get_source_output().get_node_shared_ptr());
    IE_ASSERT(const_axis != nullptr);

    
    auto axis = const_axis->cast_vector<size_t>();
    acrossSpatial = !(axis.size() == 1 && axis[0] == 1);
    channelShared = 1;

    // auto acrossSpatial = acrossSpatial; // layer->GetParamAsInt("across_spatial", 0);
    // auto channelShared = channelShared; // layer->GetParamAsInt("channel_shared", 0);
    float eps = normalize->get_eps();
    const auto weightsNode = node->input_value(1).get_node_shared_ptr();
    Data weightsBlob;
    std::tie(weightsBlob, std::ignore) = getWeightsAndBiases(model, normalize->get_friendly_name(), weightsNode, NodePtr());

    IE_ASSERT(weightsBlob != nullptr);

    auto output = outputs[0];

    auto scales = model->addConstData(normalize->get_friendly_name() + "@scales", weightsBlob->desc());

    auto stage = model->addNewStage<NormalizeStage>(normalize->get_friendly_name(), StageType::Normalize, normalize, {inputs[0], scales}, outputs);
    stage->attrs().set<bool>("acrossSpatial", acrossSpatial);
    stage->attrs().set<bool>("channelShared", channelShared);
    stage->attrs().set<float>("eps", eps);
}



}  // namespace vpu
