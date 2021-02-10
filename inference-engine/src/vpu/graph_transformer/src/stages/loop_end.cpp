// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/stages/iteration_rule.hpp"

#include <memory>
#include <string>

namespace vpu {

namespace {

class LoopEnd : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<LoopEnd>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto& endCopies = attrs().getOrDefault<IterationComponents>("end-iteration-components", {});
        for (const auto& iteration : endCopies) {
            const auto& dstIdx = static_cast<int>(iteration.first.first);
            const auto& srcIdx = static_cast<int>(iteration.second);
            orderInfo.setOutput(outputEdge(dstIdx), inputEdge(srcIdx)->input()->desc().dimsOrder());
        }

        for (const auto& outputEdge : outputEdges()) {
            const auto& attrs = outputEdge->output()->attrs();
            if (!attrs.has("end-shared-allocation")) {
                continue;
            }
            auto input = attrs.get<Data>("end-shared-allocation");
            orderInfo.setOutput(outputEdge, input->desc().dimsOrder());
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto iterationsCount = static_cast<std::int32_t>(attrs().getOrDefault<std::uint32_t>("iterations-count", g_dynamicIterationCount));
        serializer.append(iterationsCount);

        const auto& endCopies = attrs().getOrDefault<IterationComponents>("end-iteration-components", {});
        serializer.append(checked_cast<uint32_t>(endCopies.size()));

        if (attrs().has("batchId")) {
            const auto batchId = attrs().get<uint32_t>("batchId");
            const auto numDims = outputEdge(batchId)->output()->desc().numDims();
            const auto batchDimInd = numDims - 1 - dimToIeInd(Dim::N, numDims);
            serializer.append(static_cast<uint32_t>(batchDimInd));
        }

        for (const auto& component : endCopies) {
            const auto& rule = component.first.second;
            auto axis = rule.axis;
            auto axisInd = static_cast<int32_t>(output(static_cast<int>(component.first.first))->desc().dimsOrder().dimInd(axis));

            serializer.append(axisInd);
            serializer.append(rule.start);
            serializer.append(rule.stride);
            serializer.append(rule.end);
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        const auto& endCopies = attrs().getOrDefault<IterationComponents>("end-iteration-components", {});

        if (attrs().has("batchId")) {
            auto batchId = attrs().get<uint32_t>("batchId");
            outputEdge(batchId)->output()->serializeBuffer(serializer);
        }

        for (const auto& iteration : endCopies) {
            output(static_cast<int>(iteration.first.first))->serializeBuffer(serializer);
            input(static_cast<int>(iteration.second))->serializeBuffer(serializer);
        }
    }
};

}  // namespace

Stage StageBuilder::addLoopEndStage(const Model& model, const std::string& name, const DataVector& inputs, const DataVector& outputs) {
    return model->addNewStage<LoopEnd>(name, StageType::LoopEnd, nullptr, inputs, outputs);
}

}  // namespace vpu
