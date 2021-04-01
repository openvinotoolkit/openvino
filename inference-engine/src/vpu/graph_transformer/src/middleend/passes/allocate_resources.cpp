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

AllocationResult runAllocator(const Model& model, EnableShapeAllocation enableShapeAllocation, CheckOnlyCMX checkOnlyCmx) {
    VPU_PROFILE(runAllocator);

    auto& allocator = model->getAllocator();

    //
    // Clear previous allocation.
    //

    allocator.reset();

    //
    // Allocate Const/Input/Output datas.
    //

    if (checkOnlyCmx == CheckOnlyCMX::NO) {
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

        const auto allocateStageOutputs = [checkOnlyCmx, &allocator](const Stage& stage) -> AllocationResult {
            for (const auto& output : stage->outputs()) {
                if (checkOnlyCmx == CheckOnlyCMX::YES && output->memReqs() != MemoryType::CMX) {
                    continue;
                }

                if (!allocator.allocateData(output)) {
                    if (output->memReqs() == MemoryType::CMX && checkOnlyCmx == CheckOnlyCMX::NO) {
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

        if (checkOnlyCmx == CheckOnlyCMX::NO) {
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
            if (checkOnlyCmx == CheckOnlyCMX::YES && input->memReqs() != MemoryType::CMX) {
                continue;
            }

            allocator.freeData(input);
        }

        //
        // Release stage temporary buffers.
        //

        if (checkOnlyCmx == CheckOnlyCMX::NO) {
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
    // Clean up undeallocated shapes
    //

    for (auto data : model->datas()) {
        if (data->usage() != DataUsage::Output && data->usage() != DataUsage::Input) {
            continue;
        }

        if (const auto& parentEdge = data->parentDataToShapeEdge()) {
            const auto& parent = parentEdge->parent();
            if (parent->usage() == DataUsage::Intermediate && (checkOnlyCmx == CheckOnlyCMX::NO || parent->memReqs() == MemoryType::CMX)) {
                allocator.freeData(parent);
            }
        }
    }

    //
    // Allocate shape for all datas
    //

    if (enableShapeAllocation == EnableShapeAllocation::YES) {
        const auto& allocateShape = [&allocator](const Data& data) {
            if (!data->isShapeAllocated()) {
                const auto shapeLocation = allocator.allocateShape(data);
                data->setShapeAllocationInfo(shapeLocation);
            }
        };

        for (const auto& stage : model->getStages()) {
            for (const auto& input : stage->inputs()) {
                allocateShape(input);
            }
            for (const auto& output : stage->outputs()) {
                allocateShape(output);
            }
            for (const auto& tempBuffer : stage->tempBuffers()) {
                allocateShape(tempBuffer);
            }
        }

        // Allocate shape for unused inputs
        DataVector unusedInputs;
        const auto& dataObjects = model->datas();
        std::copy_if(dataObjects.begin(), dataObjects.end(), std::back_inserter(unusedInputs), [](const Data& data) {
            return data->usage() == DataUsage::Input && !data->isConsumed();
        });
        // There is no guarantee that model->datas() always contain data objects in the same order from run to run,
        // so to stabilize allocation, and as a result, the final blob, we need to sort them
        std::sort(unusedInputs.begin(), unusedInputs.end(), [](const Data& lhs, const Data& rhs) { return lhs->name() < rhs->name(); });
        std::for_each(unusedInputs.begin(), unusedInputs.end(), [&allocateShape](const Data& unusedInput) { allocateShape(unusedInput); });

        for (const auto& data : model->datas()) {
            VPU_THROW_UNLESS(data->isShapeAllocated(), "Shape for data {} with usage {} is not allocated",
                data->name(), data->usage());
        }
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

    auto allocRes = runAllocator(model, EnableShapeAllocation::YES);
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
