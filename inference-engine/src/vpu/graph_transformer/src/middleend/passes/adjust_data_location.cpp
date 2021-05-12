// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <algorithm>
#include <queue>
#include <set>
#include <memory>
#include <string>

#include <vpu/compile_env.hpp>
#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/middleend/hw/utility.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    void copyHwNetOutputs(const Model& model);
    void collectMemReqs(const Model& model);
    void resetStageOrder(const Model& model);
    void allocNonIntermediateData(const Model& model);
    void adjustModelForMemReqs(const Model& model);
    void copyHwMisalignedInput(const Model& model);
    void packDataInCmx(const Model& model);

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(adjustDataLocation);

    const auto& env = CompileEnv::get();

    copyHwNetOutputs(model);
    collectMemReqs(model);
    resetStageOrder(model);
    allocNonIntermediateData(model);
    adjustModelForMemReqs(model);
    copyHwMisalignedInput(model);
    if (env.config.packDataInCmx.getOrDefault(true)) {
        packDataInCmx(model);
    }
}

//
// Add Copy if HW operation writes to network outputs
//

void PassImpl::copyHwNetOutputs(const Model& model) {
    VPU_PROFILE(copyHwNetOutputs);

    const auto& env = CompileEnv::get();

    env.log->trace("Copy HW network outputs");
    VPU_LOGGER_SECTION(env.log);

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW) {
            continue;
        }

        auto output = stage->output(0);

        if (output->getTopParentData()->usage() == DataUsage::Output) {
            env.log->trace("HW Stage [%s] output [%s]", stage->name(), output->name());

            auto newOutput = model->duplicateData(
                output,
                "@intermediate");

            model->replaceStageOutput(stage->outputEdge(0), newOutput);

            newOutput->updateRequiredStrides(stage->getDataStridesRequirements().getOutput(stage->outputEdge(0)));

            _stageBuilder->addCopyStage(
                model,
                stage->name() + "@flush-output",
                stage->origLayer(),
                newOutput,
                output,
                "copyHwNetOutputs");
        }
    }
}

//
// Collect datas memory requirements
//

namespace {

inline void setDataMemReqs(const Data& data, MemoryType memType) {
    loopOverData(data->getTopParentData(), [memType](const Data& subData) {
        subData->setMemReqs(memType);
        return DataLoopStatus::NextChild;
    });
}

}  // namespace

void PassImpl::collectMemReqs(const Model& model) {
    VPU_PROFILE(collectMemReqs);

    const auto& env = CompileEnv::get();

    env.log->trace("Collect memory requirements");
    VPU_LOGGER_SECTION(env.log);

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW) {
            continue;
        }

        auto hwInput = stage->input(0);
        auto hwOutput = stage->output(0);

        //
        // Pattern matching
        //

        if (stage->attrs().get<HwOpType>("hwOpType") == HwOpType::CONV &&
            stage->attrs().get<int>("kernelSizeX") == 3 && stage->attrs().get<int>("kernelSizeY") == 3 &&
            stage->attrs().get<int>("kernelStride") == 1 &&
            stage->attrs().get<HwPaddingInfo>("pad").enable &&
            hwInput->desc().dim(Dim::W) == 13 && hwInput->desc().dim(Dim::H) == 13 && hwInput->desc().dim(Dim::C) == 352 &&
            hwOutput->desc().dim(Dim::C) == 1024) {
            env.log->trace("[Pattern Matching] Use CMX for HW Stage [%s] input [%s]", stage->name(), hwInput->name());

            setDataMemReqs(hwInput, MemoryType::CMX);
        } else {
            env.log->trace("[Default behavior] Use CMX for HW Stage [%s] output [%s]", stage->name(), hwOutput->name());

            setDataMemReqs(hwOutput, MemoryType::CMX);
        }
    }
}

void PassImpl::resetStageOrder(const Model& model) {
    if (!CompileEnv::get().config.hwOptimization)
        return;

    static const std::string s_expectCMXOutput {"expectCMXOutput"};

    for (const auto& stage : model->getStages()) {
        if (stage->numOutputs() == 1 && stage->output(0)->memReqs() == MemoryType::CMX) {
            stage->attrs().set(s_expectCMXOutput, true);
        }
    }
    /*
     * Add an heuristic for allocation order: when data have a split or data with several consumers,
     * walk through DDR branch before CMX branch.
     * Empirically established that there will be less attempts to copy from CMX to DDR in graph.
     */
    model->reorderStages([](const Stage& left, const Stage& right) {
        const bool leftExpectCMX  = left->attrs().getOrDefault(s_expectCMXOutput, false);
        const bool rightExpectCMX = right->attrs().getOrDefault(s_expectCMXOutput, false);
        if (leftExpectCMX != rightExpectCMX) {
            return rightExpectCMX;
        }
        return left->id() < right->id();
    });
}

//
// Allocate Const/Input/Output datas
//

void PassImpl::allocNonIntermediateData(const Model& model) {
    VPU_PROFILE(allocNonIntermediateData);

    const auto& env = CompileEnv::get();

    env.log->trace("Allocate Const/Input/Output datas");
    VPU_LOGGER_SECTION(env.log);

    auto& allocator = model->getAllocator();

    auto preprocessRes = allocator.preprocess(model);
    IE_ASSERT(preprocessRes.status == AllocationStatus::OK);
}

//
// Analyse the network from the beginning several times,
// until we satisfy all requirements or we found that it can't be done
//

void PassImpl::adjustModelForMemReqs(const Model& model) {
    VPU_PROFILE(adjustModelForMemReqs);

    const auto& env = CompileEnv::get();

    env.log->trace("Adjust Model for memory requirements");
    VPU_LOGGER_SECTION(env.log);

    auto& allocator = model->getAllocator();

    StageSet allFailedStages;

    for (;;) {
        auto allocRes = runAllocator(model);
        if (allocRes.status == AllocationStatus::OK)
            break;

        const auto failedStage = allocRes.failedStage;
        IE_ASSERT(failedStage != nullptr);

        const auto failedStageInd = failedStage->index();
        IE_ASSERT(failedStageInd >= 0);

        env.log->trace("Stage # %d [%s] failed to allocate : %s", failedStageInd, failedStage->name(), allocRes.status);
        VPU_LOGGER_SECTION(env.log);

        VPU_INTERNAL_CHECK(allFailedStages.count(failedStage) == 0,
            "Memory allocation failed: unable to satisfy requirements for stage %v with type %v",
            failedStage->name(), failedStage->type());
        allFailedStages.emplace(failedStage);

        //
        // Try to flush Data allocated in CMX
        //

        const auto failedData = allocRes.failedData;
        VPU_THROW_UNLESS(!failedData || failedData->memReqs() == MemoryType::CMX,
            R"(Request {} bytes in {} for output "{}" failed for stage "{}" of type "{}")",
            calcAllocationSize(failedData), failedData->memReqs(), failedData->name(), failedStage->name(), failedStage->type());

        auto allCmxDatas = allocator.getAllocatedDatas(MemoryType::CMX);
        env.log->trace("Got %d datas in CMX : %v", allCmxDatas.size(), allCmxDatas);

        if (allCmxDatas.empty()) {
            if (allocRes.status == AllocationStatus::SHAVES_FAILED) {
                VPU_THROW_FORMAT("Can't allocate SHAVEs for Stage node %v", failedStage->name());
            } else {
                VPU_THROW_FORMAT("Can't satisfy Data node location requirements for Stage node %s", failedStage->name());
            }
        }

        StageInputVector cmxConsumerEdges;
        cmxConsumerEdges.reserve(allCmxDatas.size() * 4);

        for (const auto& cmxData : allCmxDatas) {
            IE_ASSERT(cmxData->usage() == DataUsage::Intermediate);
            IE_ASSERT(cmxData->parentDataToDataEdge() == nullptr);

            auto cmxDataProducer = cmxData->producer();
            IE_ASSERT(cmxDataProducer != nullptr);

            auto cmxDataProducerInd = cmxDataProducer->index();
            IE_ASSERT(cmxDataProducerInd >= 0);
            IE_ASSERT(cmxDataProducerInd < failedStageInd);

            IE_ASSERT(cmxData->numConsumers() > 0);

            for (const auto& consumerEdge : cmxData->consumerEdges()) {
                if (consumerEdge->consumer()->attrs().getOrDefault<bool>("CMX-to-DDR", false)) {
                    continue;
                }

                cmxConsumerEdges.emplace_back(consumerEdge);
            }
        }

        IE_ASSERT(!cmxConsumerEdges.empty());

        for (const auto& cmxConsumerEdge : cmxConsumerEdges) {
            VPU_LOGGER_SECTION(env.log);

            auto cmxData = cmxConsumerEdge->input();
            auto cmxConsumer = cmxConsumerEdge->consumer();

            env.log->trace("CMX data [%s] consumer [%s]", cmxData->name(), cmxConsumer->name());
            VPU_LOGGER_SECTION(env.log);

            Data ddrCopy;
            if (cmxData->attrs().has("ddrCopy")) {
                env.log->trace("It already has DDR spill");

                ddrCopy = cmxData->attrs().get<Data>("ddrCopy");
            } else {
                env.log->trace("Create new DDR spill");

                ddrCopy = model->duplicateData(
                    cmxData,
                    formatString("@DDR-copy"));

                ddrCopy->updateRequiredStrides(cmxData->requiredStrides());
                ddrCopy->setMemReqs(MemoryType::DDR);

                auto copyStage = _stageBuilder->addCopyStage(
                    model,
                    formatString("%s@move-to-DDR", cmxData->name()),
                    failedStage->origLayer(),
                    cmxData,
                    ddrCopy,
                    "CopyFromCMXToDDRAdjustment");
                copyStage->attrs().set<bool>("CMX-to-DDR", true);

                cmxData->attrs().set("ddrCopy", ddrCopy);
            }

            model->replaceStageInput(cmxConsumerEdge, ddrCopy);

            env.log->trace("Update child datas");
            for (const auto& childDataEdge : cmxData->childDataToDataEdges()) {
                VPU_LOGGER_SECTION(env.log);

                auto order = childDataEdge->order();

                if (childDataEdge->connectionMode() == SharedConnectionMode::SINGLE_STAGE && order == SharedDataOrder::ParentWritesToChild &&
                    childDataEdge->connection() == cmxConsumer) {
                    auto childData = childDataEdge->child();

                    auto mode = childDataEdge->mode();
                    auto offset = childDataEdge->attrs().getOrDefault<DimValues>("offset", DimValues());

                    env.log->trace("Child data [%s] : mode [%v] offset [%v]", childData->name(), mode, offset);

                    model->replaceDataToDataParent(childDataEdge, ddrCopy);

                    loopOverData(childData, [](const Data& subData) {
                        subData->setMemReqs(MemoryType::DDR);
                        return DataLoopStatus::NextChild;
                    });
                }
            }
        }
    }
}

//
// Add Copy if HW operation reads misaligned input
//

void PassImpl::copyHwMisalignedInput(const Model& model) {
    VPU_PROFILE(copyHwMisalignedInput);

    const auto& env = CompileEnv::get();

    env.log->trace("Add Copy for misaligned HW inputs");
    VPU_LOGGER_SECTION(env.log);

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW) {
            continue;
        }

        auto inputEdge = stage->inputEdge(0);
        auto input = inputEdge->input();
        IE_ASSERT(input->dataLocation().location != Location::None);

        if (input->dataLocation().offset % 16 != 0) {
            env.log->trace("HW Stage [%s] input [%s]", stage->name(), input->name());

            auto newInput = model->duplicateData(
                input,
                "@aligned-ptr");
            newInput->updateRequiredStrides(input->requiredStrides());
            newInput->setMemReqs(MemoryType::DDR);

            _stageBuilder->addCopyStage(
                model,
                stage->name() + "@align-input-ptr",
                stage->origLayer(),
                input,
                newInput,
                "copyHwMisalignedInput");

            model->replaceStageInput(stage->inputEdge(0), newInput);
        }
    }
}

//
// Try to put HW inputs to CMX if possible
//

void PassImpl::packDataInCmx(const Model& model) {
    VPU_PROFILE(packDataInCmx);

    const auto& env = CompileEnv::get();

    env.log->trace("Try to use CMX for HW inputs");
    VPU_LOGGER_SECTION(env.log);

    auto& allocator = model->getAllocator();

    //
    // Collect candidates
    //

    std::queue<Data> candidatesForCMX;

    auto& visitedDatas = allocator.getCandidatesForCMX();
    visitedDatas.clear();

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW)
            continue;

        for (const auto& input : stage->inputs()) {
            auto topParent = input->getTopParentData();

            if (topParent->usage() != DataUsage::Intermediate)
                continue;

            auto topParentMemType = topParent->memReqs();
            if (topParentMemType == MemoryType::CMX)
                continue;

            auto producer = input->producer();
            IE_ASSERT(producer != nullptr);

            if (producer->type() == StageType::Copy &&
                producer->attrs().getOrDefault<bool>("CMX-to-DDR", false)) {
                continue;
            }

            if (producer->getSHAVEsRequirements() != StageSHAVEsRequirements::NeedMax) {
                if (visitedDatas.count(topParent) == 0) {
                    candidatesForCMX.push(topParent);
                    visitedDatas.insert(topParent);
                }
            }
        }
    }

    //
    // Try candidates one by one -> if allocation cycle is successfull, leave the data in CMX
    //

    while (!candidatesForCMX.empty()) {
        auto curCandidate = candidatesForCMX.front();
        candidatesForCMX.pop();

        env.log->trace("Try use CMX for Data [%s]", curCandidate->name());
        VPU_LOGGER_SECTION(env.log);

        IE_ASSERT(curCandidate->parentDataToDataEdge() == nullptr);
        IE_ASSERT(curCandidate->usage() == DataUsage::Intermediate);

        auto curMemoryType = curCandidate->memReqs();
        IE_ASSERT(curMemoryType == MemoryType::DDR);

        loopOverData(curCandidate, [](const Data& subData) {
            subData->setMemReqs(MemoryType::CMX);
            return DataLoopStatus::NextChild;
        });

        auto allocRes = runAllocator(model, EnableShapeAllocation::NO, CheckOnlyCMX::YES);
        env.log->trace("Allocation result : %v", allocRes.status);

        if (allocRes.status != AllocationStatus::OK) {
            env.log->trace("Revert CMX usage for Data [%s]", curCandidate->name());

            loopOverData(curCandidate, [](const Data& subData) {
                subData->setMemReqs(MemoryType::DDR);
                return DataLoopStatus::NextChild;
            });
        }
    }
}

}  // namespace

Pass::Ptr PassManager::adjustDataLocation() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
