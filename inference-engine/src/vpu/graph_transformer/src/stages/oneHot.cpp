// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <precision_utils.h>
#include <memory>
#include <set>

namespace vpu {
namespace {
class OneHot final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<OneHot>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();
        orderInfo.setInput(inputEdge(0), DimsOrder::fromNumDims(input->desc().numDims()));
        orderInfo.setOutput(outputEdge(0), DimsOrder::fromNumDims(output->desc().numDims()));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
                                 {{DataType::S32}},
                                 {{DataType::FP16}});
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto axis = attrs().get<int>("axis");
        auto depth = attrs().get<unsigned int>("depth");
        auto on_value = attrs().get<float>("on_value");
        auto off_value = attrs().get<float>("off_value");

        serializer.append(static_cast<int32_t>(axis));
        serializer.append(static_cast<uint32_t>(depth));
        serializer.append(on_value);
        serializer.append(off_value);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(inputEdges().size() == 1);
        IE_ASSERT(outputEdges().size() == 1);

        auto input = inputEdges()[0]->input();
        auto output = outputEdges()[0]->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseOneHot(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto oneHot = ngraph::as_type_ptr<ngraph::op::v1::OneHot>(node);
    VPU_THROW_UNLESS(oneHot != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    const auto depthNode = std::dynamic_pointer_cast<ngraph::opset1::Constant>(oneHot->input_value(1).get_node_shared_ptr());
    const auto onValueNode = std::dynamic_pointer_cast<ngraph::opset1::Constant>(oneHot->input_value(2).get_node_shared_ptr());
    const auto offValueNode = std::dynamic_pointer_cast<ngraph::opset1::Constant>(oneHot->input_value(3).get_node_shared_ptr());
    VPU_THROW_UNLESS(depthNode != nullptr && onValueNode != nullptr && offValueNode != nullptr,
                    "Can't parse node with name %s and type %s. Can't get params", node->get_friendly_name(), node->get_type_name());

    int axis = oneHot->get_axis() == -1 ? 0 : inputs[0]->desc().numDims() - oneHot->get_axis();
    auto depthValue = std::stoi(depthNode->convert_value_to_string(0));
    auto onValue = std::stof(onValueNode->convert_value_to_string(0));
    auto offValue = std::stof(offValueNode->convert_value_to_string(0));

    auto stage = model->addNewStage<OneHot>(oneHot->get_name(), StageType::OneHot, oneHot, inputs, outputs);

    stage->attrs().set<int>("axis", axis);
    stage->attrs().set<unsigned int>("depth", depthValue);
    stage->attrs().set<float>("on_value", onValue);
    stage->attrs().set<float>("off_value", offValue);
}

}  // namespace vpu
