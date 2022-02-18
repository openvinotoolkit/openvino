// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <cmath>

#include <vector>
#include <unordered_set>
#include <memory>
#include <utility>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model& model) override;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(swapConcatAndHwOps);

    for (const auto& concatStage : model->getStages()) {
        if (concatStage == nullptr)
            continue;

        if (concatStage->type() != StageType::StubConcat)
            continue;

        IE_ASSERT(concatStage->numInputs() > 0);
        IE_ASSERT(concatStage->numOutputs() == 1);

        auto concatOutput = concatStage->output(0);

        //
        // Check concat axis
        //

        // TODO: other cases?
        auto concatAxis = concatStage->attrs().getOrDefault<Dim>("axis", Dim::Invalid);
        if (concatAxis != Dim::C) {
            continue;
        }

        //
        // All concat inputs must be used by concat only
        //

        bool concatIsTheOnlyConsumer = true;
        for (const auto& concatInput : concatStage->inputs()) {
            if (concatInput->numConsumers() != 1) {
                concatIsTheOnlyConsumer = false;
                break;
            }
        }
        if (!concatIsTheOnlyConsumer) {
            continue;
        }

        //
        // Collect next stages (HW Pool and ReLU)
        //

        StageVector nextStages;
        nextStages.reserve(2);

        for (auto curOutput = concatOutput;;) {
            if (curOutput->usage() != DataUsage::Intermediate) {
                break;
            }

            if (curOutput->numConsumers() != 1) {
                break;
            }

            auto curConsumer = curOutput->singleConsumer();
            auto curConsumerHW = curConsumer->attrs().getOrDefault<bool>("tryHW", false);

            if (curConsumer->type() == StageType::StubMaxPool && curConsumerHW) {
                // OK
            } else if (curConsumer->type() == StageType::Relu ||
                       curConsumer->type() == StageType::LeakyRelu) {
                // OK
            } else {
                break;
            }

            nextStages.emplace_back(curConsumer);

            curOutput = curConsumer->output(0);
        }

        if (nextStages.empty())
            continue;

        //
        // Swap next stages and concat
        //

        auto lastInputs = concatStage->inputs() | asSmallVector();
        auto lastOutput = concatOutput;

        for (const auto& nextStage : nextStages) {
            auto nextOutput = nextStage->output(0);

            model->disconnectStage(nextStage);

            DataVector newOutputs;
            newOutputs.reserve(lastInputs.size());

            int subInd = 0;
            for (const auto& curInput : lastInputs) {
                auto postfix = formatString("@sub=%d/%d", subInd + 1, lastInputs.size());

                auto newDesc = nextOutput->desc();
                newDesc.setDim(Dim::C, curInput->desc().dim(Dim::C));

                auto newOutput = model->duplicateData(
                    nextOutput,
                    postfix,
                    newDesc);

                model->duplicateStage(
                    nextStage,
                    postfix,
                    {curInput},
                    {newOutput});

                newOutputs.emplace_back(std::move(newOutput));

                ++subInd;
            }

            model->removeStage(nextStage);

            lastInputs.swap(newOutputs);
            lastOutput = nextOutput;
        }

        for (const auto& inEdge : concatStage->inputEdges()) {
            model->replaceStageInput(inEdge, lastInputs.at(inEdge->portInd()));
        }
        model->replaceStageOutput(concatStage->outputEdge(0), lastOutput);
    }
}

}  // namespace

Pass::Ptr PassManager::swapConcatAndHwOps() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
