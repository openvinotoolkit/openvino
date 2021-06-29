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

void FrontEnd::parseCopy(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    _stageBuilder->addCopyStage(model, node->get_name(), node, inputs[0], outputs[0], "parseCopy");
}

namespace {

class CopyStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<CopyStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement().remove(0));
        stridesInfo.setOutput(outputEdge(0), StridesRequirement().remove(0));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void initialCheckImpl() const override {
        const auto& type = input(0)->desc().type();
        assertInputsOutputsTypes(this, {{type}}, {{type}});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addCopyStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        const Data& input,
        const Data& output,
        const std::string& origin) {
    Stage copyStage = model->addNewStage<CopyStage>(
        name,
        StageType::Copy,
        node,
        {input},
        {output});
    copyStage->attrs().set<std::string>("origin", origin);
    return copyStage;
}

}  // namespace vpu
