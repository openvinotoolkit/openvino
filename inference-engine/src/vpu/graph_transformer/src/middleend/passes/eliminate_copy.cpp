// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <unordered_set>
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <queue>

#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) :
            _stageBuilder(stageBuilder) {
    }

    void run(const Model& model) override;

private:
    static bool isApplicable(const Stage& copyStage);

private:
    StageBuilder::Ptr _stageBuilder;
};

bool PassImpl::isApplicable(const Stage& copyStage) {
    auto copyInput = copyStage->input(0);
    auto copyOutput = copyStage->output(0);

    IE_ASSERT(copyInput->usage() == DataUsage::Intermediate);
    IE_ASSERT(copyOutput->usage() == DataUsage::Intermediate);
    IE_ASSERT(copyInput->producerEdge() != nullptr);
    IE_ASSERT(copyInput->desc().dimsOrder() == copyOutput->desc().dimsOrder());

    if (copyInput->parentDataEdge() != nullptr) {
        return false;
    }
    if (copyInput->numChildDatas() > 0) {
        return false;
    }

    if (!checkStrides(copyInput->desc(), copyOutput->strides(), copyInput->requiredStrides())) {
        return false;
    }
    if (!checkStrides(copyOutput->desc(), copyInput->strides(), copyOutput->requiredStrides())) {
        return false;
    }

    auto copyOutputTopParent = copyOutput->getTopParentData();
    if (copyOutputTopParent->usage() != DataUsage::Intermediate) {
        return false;
    }

    IE_ASSERT(copyOutput->numConsumers() == 1);

    auto specialConsumer = copyOutput->singleConsumer();
    IE_ASSERT(specialConsumer->category() == StageCategory::Special);

    return true;
}

void PassImpl::run(const Model& model) {
    VPU_PROFILE(eliminateCopyStages);

    const int nMaxCopyStages = 23000;
    const auto& env = CompileEnv::get();

    std::queue<Stage> copyToRemove;

    if (!env.config.copyOptimization.hasValue()) {
        int nCopyStages = 0;
        for (const auto& stage : model->getStages()) {
            if (stage->type() == StageType::Copy) {
                ++nCopyStages;
            }
        }

        // Eliminate copy will take more than an hour in that case
        if (nCopyStages > nMaxCopyStages) {
            env.log->warning(
                "Pass [eliminateCopyStages] SKIPPED : number of copy stages (%d) is larger than threshold %d",
                nCopyStages, nMaxCopyStages);
            return;
        }
    }

    for (const auto& copyStage : model->getStages()) {
        if (copyStage->type() != StageType::Copy) {
            continue;
        }

        auto isOptional = copyStage->attrs().getOrDefault<bool>("optional", false);
        if (!isOptional) {
            continue;
        }

        if (isApplicable(copyStage)) {
            copyToRemove.push(copyStage);
        }
    }

    while (!copyToRemove.empty()) {
        auto copyStage = copyToRemove.front();
        copyToRemove.pop();

        if (!isApplicable(copyStage)) {
            continue;
        }

        auto copyInput = copyStage->input(0);
        auto copyOutput = copyStage->output(0);

        auto copyOutputTopParent = copyOutput->getTopParentData();

        auto copyStageName = copyStage->name();
        auto copyOrigLayer = copyStage->origLayer();

        auto copyProducer = copyInput->producer();
        auto specialConsumer = copyOutput->singleConsumer();

        //
        // Try to remove Copy and redirect (copyProducer) to [copyOutput] and ask CMX location for it.
        // Run allocation and if it fails -> revert the changes in the Model.
        //

        model->removeStage(copyStage);

        auto oldMemoryType = copyOutputTopParent->memReqs();
#ifndef NDEBUG
        loopOverData(copyOutputTopParent, [oldMemoryType](const Data& subData) {
            auto subMemType = subData->memReqs();
            IE_ASSERT(subMemType == oldMemoryType);
            return DataLoopStatus::NextChild;
        });
#endif
        if (oldMemoryType != MemoryType::CMX) {
            loopOverData(copyOutputTopParent, [](const Data& subData) {
                subData->setMemReqs(MemoryType::CMX);
                return DataLoopStatus::NextChild;
            });
        }

        model->replaceStageOutput(copyProducer->outputEdge(0), copyOutput);

        StageInputVector prevEdges;
        prevEdges.reserve(copyInput->numConsumers());
        for (const auto& consumerEdge : copyInput->consumerEdges()) {
            prevEdges.emplace_back(consumerEdge);
            model->replaceStageInput(consumerEdge, copyOutput);
        }

        auto allocRes = runAllocator(model, true);
        if (allocRes.status != AllocationStatus::OK) {
            model->replaceStageOutput(copyProducer->outputEdge(0), copyInput);

            for (const auto& p : prevEdges) {
                model->replaceStageInput(p, copyInput);
            }

            _stageBuilder->addCopyStage(model, copyStageName, copyOrigLayer, copyInput, copyOutput, "Non-eliminated copy");

            loopOverData(copyOutputTopParent, [oldMemoryType](const Data& subData) {
                subData->setMemReqs(oldMemoryType);
                return DataLoopStatus::NextChild;
            });
        }
    }
}

}  // namespace

Pass::Ptr PassManager::eliminateCopyStages() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
