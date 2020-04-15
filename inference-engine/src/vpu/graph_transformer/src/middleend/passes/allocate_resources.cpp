// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <unordered_map>
#include <list>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <vector>
#include <string>
#include <set>
#include <queue>
#include <memory>

#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/utils/auto_scope.hpp>

namespace vpu {

//
// runAllocator
//

AllocationResult runAllocator(const Model& model, bool onlyCheckCMX) {
    VPU_PROFILE(runAllocator);

    auto& allocator = model->getAllocator();

    //
    // Clear previous allocation.
    //

    allocator.reset();

    //
    // Allocate Const/Input/Output datas.
    //

    if (!onlyCheckCMX) {
        auto result = allocator.preprocess(model);
        if (result.status != vpu::AllocationStatus::OK) {
            return result;
        }
    }

    //
    // Allocate resources per stage.
    //

    for (const auto& stage : model->getStages()) {
        //
        // Release SHAVEs in any case at the end of iteration.
        //

        stage->setNumSHAVEs(0);
        AutoScope scope([&allocator]() {
            allocator.getAllocatorOfShaves().freeSHAVEs();
        });

        //
        // Get stage SHAVE requirements.
        //

        auto reqs = stage->getSHAVEsRequirements();

        //
        // Allocate SHAVEs for NeedMax before the Data allocation.
        //

        if (reqs == StageSHAVEsRequirements::NeedMax) {
            if (!allocator.getAllocatorOfShaves().allocateSHAVEs(stage, reqs)) {
                allocator.setNeedToAllocNonIntermData();

                AllocationResult res;
                res.status = AllocationStatus::SHAVES_FAILED;
                res.failedStage = stage;
                return res;
            }
        }

        //
        // Allocate stage outputs.
        //

        const auto allocateStageOutputs = [onlyCheckCMX, &allocator](const Stage& stage) -> AllocationResult {
            for (const auto& output : stage->outputs()) {
                if (onlyCheckCMX && output->memReqs() != MemoryType::CMX) {
                    continue;
                }

                if (!allocator.allocateData(output)) {
                    if (output->memReqs() == MemoryType::CMX && !onlyCheckCMX) {
                        if (allocator.removeCMXCandidates(output)) {
                            if (allocator.allocateData(output)) {
                                continue;
                            }
                        }

                        allocator.setNeedToAllocNonIntermData();
                    }

                    AllocationResult res;
                    res.status = AllocationStatus::DATA_FAILED;
                    res.failedStage = stage;
                    res.failedData = output;
                    return res;
                }
            }

            return AllocationResult();
        };

        const auto outputsAllocationStatus = allocateStageOutputs(stage);
        if (outputsAllocationStatus.status != AllocationStatus::OK) {
            return outputsAllocationStatus;
        }

        //
        // Allocate stage temporary buffers.
        //

        if (!onlyCheckCMX) {
            for (const auto& tempBufferEdge : stage->tempBufferEdges()) {
                if (!allocator.allocateData(tempBufferEdge->tempBuffer())) {
                    allocator.setNeedToAllocNonIntermData();

                    AllocationResult res;
                    res.status = AllocationStatus::DATA_FAILED;
                    res.failedStage = stage;
                    res.failedData = tempBufferEdge->tempBuffer();
                    return res;
                }
            }
        }

        //
        // Allocate limited SHAVEs after the Data allocation.
        //

        if (reqs != StageSHAVEsRequirements::NeedMax) {
            if (!allocator.getAllocatorOfShaves().allocateSHAVEs(stage, reqs)) {
                allocator.setNeedToAllocNonIntermData();

                AllocationResult res;
                res.status = AllocationStatus::SHAVES_FAILED;
                res.failedStage = stage;
                return res;
            }
        }

        //
        // Release stage inputs.
        //

        for (const auto& input : stage->inputs()) {
            if (onlyCheckCMX && input->memReqs() != MemoryType::CMX) {
                continue;
            }

            allocator.freeData(input);
        }

        //
        // Release stage temporary buffers.
        //

        if (!onlyCheckCMX) {
            for (const auto& tempBufferEdge : stage->tempBufferEdges()) {
                allocator.freeData(tempBufferEdge->tempBuffer());
            }
        }

        if (stage->type() == StageType::LoopStart) {
            // To avoid re-usage Loop End's outputs memory by data objects inside loop - allocate it before them
            const auto& loopEnd = stage->attrs().get<Stage>("loop-end");
            const auto loopEndAllocation = allocateStageOutputs(loopEnd);
            if (loopEndAllocation.status != AllocationStatus::OK) {
                return loopEndAllocation;
            }
        }
    }

    //
    // Allocate shape for all datas
    //

    for (auto data : model->datas()) {
        const auto shapeLocation = allocator.allocateConstShape(data);
        data->setShapeAllocationInfo(shapeLocation);
    }

    return AllocationResult();
}

//
// allocateResources
//

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model& model) override;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(allocateResources);

    auto& allocator = model->getAllocator();

    //
    // Allocate all resources
    //

    auto allocRes = runAllocator(model);
    IE_ASSERT(allocRes.status == AllocationStatus::OK);

    //
    // Allocator self-check
    //

    allocator.selfCheck();

    //
    // Allocation statistics
    //

    model->attrs().set<UsedMemory>("usedMemory", allocator.usedMemoryAmount());
}

}  // namespace

Pass::Ptr PassManager::allocateResources() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
