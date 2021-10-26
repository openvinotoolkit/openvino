// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <string>
#include <memory>
#include <set>
#include <list>

#include <vpu/configuration/options/hw_inject_stages.hpp>
#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/compile_env.hpp>

#include <stack>

namespace vpu {

void printTo(std::ostream&, const std::list<Stage>::iterator&) {
}

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model& model) override;
private:
    void markModelWithLoops(const Model& model) const;
    static const char s_loopAttribute[];
};

const char PassImpl::s_loopAttribute[] = "loop";

void PassImpl::markModelWithLoops(const vpu::Model &model) const {
    std::stack<Stage> loops;
    for (const auto& stage : model->getStages()) {
        VPU_THROW_UNLESS(stage->type() != StageType::LoopEnd || !loops.empty(), R"(Incorrect graph: there is a "LoopEnd" ("{}") without paired "LoopStart")",
            stage->name());

        if (stage->type() == StageType::LoopStart) {
            loops.push(stage);
        }

        if (!loops.empty()) {
            stage->attrs().set<Stage>(s_loopAttribute, loops.top());
        }

        if (stage->type() == StageType::LoopEnd) {
            loops.pop();
        }
    }

    VPU_THROW_UNLESS(loops.empty(), R"(Incorrect graph: there is a "LoopStart" ("{}") without paired "LoopEnd")", loops.top()->name());
}

void PassImpl::run(const Model& model) {
    VPU_PROFILE(injectSw);

    const int nMaxStagesForInjectSw = 10000;
    const auto& env = CompileEnv::get();

    //
    // Collect HW and SW candidates
    //

    if (!env.config.get<HwInjectStagesOption>().hasValue() &&
        model->numStages() > nMaxStagesForInjectSw) {
        env.log->warning(
            "Pass [injectSw] SKIPPED : number of stages (%d) is larger than threshold %d",
            model->numStages(), nMaxStagesForInjectSw);
        return;
    }

    StageVector hwStages;
    std::list<Stage> swStages;
    hwStages.reserve(checked_cast<size_t>(model->numStages()));
    for (const auto& stage : model->getStages()) {
        if (stage->category() == StageCategory::HW) {
            hwStages.emplace_back(stage);
        } else if (stage->category() == StageCategory::DMA || stage->category() == StageCategory::SHAVE) {
            if (stage->getSHAVEsRequirements() != StageSHAVEsRequirements::NeedMax) {
                auto it = swStages.emplace(swStages.end(), stage);
                stage->attrs().set<std::list<Stage>::iterator>("swStagesPos", it);
            }
        }
    }

    markModelWithLoops(model);

    //
    // Try to merge HW and SW stages
    //

    StageVector swCandidates;

    for (const auto& hwStage : hwStages) {
        swCandidates.clear();

        model->buildStageOrder();

        for (const auto& swStage : swStages) {
            auto hwInd = hwStage->index();
            IE_ASSERT(hwInd >= 0);

            auto swInd = swStage->index();
            IE_ASSERT(swInd >= 0);
            IE_ASSERT(swInd != hwInd);

            const auto hwLoop = hwStage->attrs().getOrDefault<Stage>(s_loopAttribute, nullptr);
            const auto swLoop = swStage->attrs().getOrDefault<Stage>(s_loopAttribute, nullptr);
            if (hwLoop != swLoop) {
                // to be injected both stages must belong to the same loop or don't belong to any
                continue;
            }

            //
            // Check execution order
            //

            bool isOK = true;

            if (swInd > hwInd) {
                //
                // SW producer must be executed after HW stage
                //

                for (const auto& swProducer : swStage->prevStages()) {
                    auto swProducerInd = swProducer->index();
                    IE_ASSERT(swProducerInd >= 0);
                    IE_ASSERT(swProducerInd < swInd);

                    if (swProducerInd >= hwInd) {
                        isOK = false;
                        break;
                    }
                }
            } else {
                //
                // HW producer must be executed after SW stage
                //

                for (const auto& hwProducer : hwStage->prevStages()) {
                    auto hwProducerInd = hwProducer->index();
                    IE_ASSERT(hwProducerInd >= 0);
                    IE_ASSERT(hwProducerInd < hwInd);

                    if (hwProducerInd >= swInd) {
                        isOK = false;
                        break;
                    }
                }
            }

            if (isOK) {
                swCandidates.push_back(swStage);
            }
        }

        for (const auto& swStage : swCandidates) {
            //
            // Try to inject and check allocation, if it is failed -> revert
            //

            auto edge = model->injectStage()
                    .parentHW(hwStage)
                    .childSW(swStage)
                    .done();

            auto allocRes = runAllocator(model, EnableShapeAllocation::NO, CheckOnlyCMX::YES);
            if (allocRes.status == AllocationStatus::OK) {
                // TODO: try to merge more than one SW stage?
                break;
            } else {
                model->revertInjection(edge);
            }
        }

        //
        // Remove injected stages from candidates list
        //

        if (const auto injectedStage = hwStage->injectedStage()) {
            auto it = injectedStage->attrs().get<std::list<Stage>::iterator>("swStagesPos");

            IE_ASSERT(it != swStages.end());
            swStages.erase(it);

            injectedStage->attrs().erase("swStagesPos");
        }
    }
}

}  // namespace

Pass::Ptr PassManager::injectSw() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
