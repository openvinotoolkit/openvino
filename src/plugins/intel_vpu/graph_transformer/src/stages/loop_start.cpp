// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/stages/iteration_rule.hpp"

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
        const auto iterationsCount = static_cast<std::int32_t>(attrs().getOrDefault<std::uint32_t>("iterations-count", g_dynamicIterationCount));
        serializer.append(iterationsCount);
        serializer.append(attrs().get<uint32_t>("stages-count"));

        const auto& startCopies = attrs().getOrDefault<IterationComponents>("start-iteration-components", {});
        serializer.append(checked_cast<uint32_t>(startCopies.size()));

        if (attrs().has("batchId")) {
            const auto batchId = attrs().get<uint32_t>("batchId");
            const auto numDims = inputEdge(batchId)->input()->desc().numDims();
            const auto batchDimInd = numDims - 1 - dimToIeInd(Dim::N, numDims);
            serializer.append(static_cast<uint32_t>(batchDimInd));
        }

        for (const auto& component : startCopies) {
            const auto& rule = component.first.second;
            auto axis = rule.axis;
            auto axisInd = static_cast<int32_t>(input(static_cast<int>(component.first.first))->desc().dimsOrder().dimInd(axis));

            serializer.append(axisInd);
            serializer.append(rule.start);
            serializer.append(rule.stride);
            serializer.append(rule.end);
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        const auto& startCopies = attrs().getOrDefault<IterationComponents>("start-iteration-components", {});

        if (attrs().has("batchId")) {
            const auto batchId = attrs().get<uint32_t>("batchId");
            inputEdge(batchId)->input()->serializeBuffer(serializer);
        }

        for (const auto& iteration : startCopies) {
            input(static_cast<int>(iteration.first.first))->serializeBuffer(serializer);
            output(static_cast<int>(iteration.second))->serializeBuffer(serializer);
        }
    }
};

}  // namespace

Stage StageBuilder::addLoopStartStage(const Model& model, const std::string& name, const DataVector& inputs, const DataVector& outputs) {
    return model->addNewStage<LoopStart>(name, StageType::LoopStart, nullptr, inputs, outputs);
}

}  // namespace vpu
