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
            const auto& dstIdx = iteration.first.first;
            const auto& srcIdx = iteration.second;
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
        for (const auto& inputEdge : inputEdges()) {
            stridesInfo.setInput(inputEdge, StridesRequirement::compact());
        }

        for (const auto& outputEdge : outputEdges()) {
            stridesInfo.setOutput(outputEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        serializer.append(attrs().get<uint32_t>("iterations-count"));

        const auto& endCopies = attrs().getOrDefault<IterationComponents>("end-iteration-components", {});
        serializer.append(checked_cast<uint32_t>(endCopies.size()));
        for (const auto& component : endCopies) {
            const auto& rule = component.first.second;
            auto axis = rule.axis;
            auto axisInd = static_cast<int32_t>(output(component.first.first)->desc().dimsOrder().dimInd(axis));

            serializer.append(axisInd);
            serializer.append(rule.start);
            serializer.append(rule.stride);
            serializer.append(rule.end);
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        const auto& endCopies = attrs().getOrDefault<IterationComponents>("end-iteration-components", {});
        for (const auto& iteration : endCopies) {
            output(iteration.first.first)->serializeNewBuffer(serializer);
            input(iteration.second)->serializeNewBuffer(serializer);
        }
    }
};

}  // namespace

Stage StageBuilder::addLoopEndStage(const Model& model, const std::string& name, const DataVector& inputs, const DataVector& outputs) {
    return model->addNewStage<LoopEnd>(name, StageType::LoopEnd, nullptr, inputs, outputs);
}

}  // namespace vpu
