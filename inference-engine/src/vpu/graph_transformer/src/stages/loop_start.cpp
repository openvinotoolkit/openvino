// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/stages/iteration_rule.hpp"

#include <utility>
#include <map>
#include <memory>
#include <string>

namespace vpu {

namespace {

class LoopStart : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<LoopStart>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        for (int i = 0; i < numInputs(); ++i) {
            orderInfo.setOutput(outputEdge(i), inputEdge(i)->input()->desc().dimsOrder());
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) override {
    }

    void initialCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        serializer.append(attrs().get<uint32_t>("iterations-count"));
        serializer.append(attrs().get<uint32_t>("stages-count"));

        const auto& startCopies = attrs().getOrDefault<IterationComponents>("start-iteration-components", {});
        serializer.append(checked_cast<uint32_t>(startCopies.size()));
        for (const auto& component : startCopies) {
            const auto& rule = component.first.second;
            auto axis = rule.axis;
            auto axisInd = static_cast<int32_t>(input(component.first.first)->desc().dimsOrder().dimInd(axis));

            serializer.append(axisInd);
            serializer.append(rule.start);
            serializer.append(rule.stride);
            serializer.append(rule.end);
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        const auto& startCopies = attrs().getOrDefault<IterationComponents>("start-iteration-components", {});
        for (const auto& iteration : startCopies) {
            input(iteration.first.first)->serializeBuffer(serializer);
            output(iteration.second)->serializeBuffer(serializer);
        }
    }
};

}  // namespace

Stage StageBuilder::addLoopStartStage(const Model& model, const std::string& name, const DataVector& inputs, const DataVector& outputs) {
    return model->addNewStage<LoopStart>(name, StageType::LoopStart, nullptr, inputs, outputs);
}

}  // namespace vpu
