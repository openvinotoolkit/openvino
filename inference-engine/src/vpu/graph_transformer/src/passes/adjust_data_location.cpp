// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <algorithm>
#include <queue>
#include <set>
#include <memory>

#include <vpu/allocator.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/utils/extra.hpp>
#include <vpu/hw/utility.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    void copyHwNetOutputs(const Model::Ptr& model);
    void collectMemReqs(const Model::Ptr& model);
    void allocNonIntermediateData(const Model::Ptr& model);
    void adjustModelForMemReqs(const Model::Ptr& model);
    void copyHwMisalignedInput(const Model::Ptr& model);
    void packDataInCmx(const Model::Ptr& model);

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(adjustDataLocation);

    const auto& env = CompileEnv::get();

    VPU_LOGGER_SECTION(env.log);

    copyHwNetOutputs(model);
    collectMemReqs(model);
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

void PassImpl::copyHwNetOutputs(const Model::Ptr& model) {
    VPU_PROFILE(copyHwNetOutputs);

    const auto& env = CompileEnv::get();

    env.log->debug("Copy HW network outputs");
    VPU_LOGGER_SECTION(env.log);

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW) {
            continue;
        }

        auto output = stage->output(0);

        if (output->usage() == DataUsage::Output) {
            env.log->debug("HW Stage [%s] output [%s]", stage->name(), output->name());

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
                output);
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

void PassImpl::collectMemReqs(const Model::Ptr& model) {
    VPU_PROFILE(collectMemReqs);

    const auto& env = CompileEnv::get();

    env.log->debug("Collect memory requirements");
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
            env.log->debug("[Pattern Matching] Use CMX for HW Stage [%s] input [%s]", stage->name(), hwInput->name());

            setDataMemReqs(hwInput, MemoryType::CMX);
        } else {
            env.log->debug("[Default behavior] Use CMX for HW Stage [%s] output [%s]", stage->name(), hwOutput->name());

            setDataMemReqs(hwOutput, MemoryType::CMX);
        }
    }
}

//
// Allocate Const/Input/Output datas
//

void PassImpl::allocNonIntermediateData(const Model::Ptr& model) {
    VPU_PROFILE(allocNonIntermediateData);

    const auto& env = CompileEnv::get();

    env.log->debug("Allocate Const/Input/Output datas");
    VPU_LOGGER_SECTION(env.log);

    auto& allocator = model->getAllocator();

    auto preprocessRes = allocator.preprocess(model);
    IE_ASSERT(preprocessRes.status == AllocationStatus::OK);
}

//
// Analyse the network from the beginning several times,
// until we satisfy all requirements or we found that it can't be done
//

void PassImpl::adjustModelForMemReqs(const Model::Ptr& model) {
    VPU_PROFILE(adjustModelForMemReqs);

    const auto& env = CompileEnv::get();

    env.log->debug("Adjust Model for memory requirements");
    VPU_LOGGER_SECTION(env.log);

    auto& allocator = model->getAllocator();

    StageSet allFailedStages;

    for (;;) {
        auto allocRes = runAllocator(model);
        if (allocRes.status == AllocationStatus::OK)
            break;

        auto failedStage = allocRes.failedStage;
        IE_ASSERT(failedStage != nullptr);

        auto failedStageInd = failedStage->index();
        IE_ASSERT(failedStageInd >= 0);

        env.log->debug("Stage # %d [%s] failed to allocate : %s", failedStageInd, failedStage->name(), allocRes.status);
        VPU_LOGGER_SECTION(env.log);

        IE_ASSERT(allFailedStages.count(failedStage) == 0);
        allFailedStages.emplace(failedStage);

        //
        // Try to flush Data allocated in CMX
        //

        auto allCmxDatas = allocator.getAllocatedDatas(MemoryType::CMX);
        env.log->debug("Got %d datas in CMX : %v", allCmxDatas.size(), allCmxDatas);

        if (allCmxDatas.empty()) {
            if (allocRes.status == AllocationStatus::SHAVES_FAILED) {
                VPU_LOG_AND_THROW(env.log, "Can't allocate SHAVEs for stage %s", failedStage->name());
            } else {
                VPU_LOG_AND_THROW(env.log, "Can't satisfy data location requirements for stage %s", failedStage->name());
            }
        }

        StageInputVector cmxConsumerEdges;
        cmxConsumerEdges.reserve(allCmxDatas.size() * 4);

        for (const auto& cmxData : allCmxDatas) {
            IE_ASSERT(cmxData->usage() == DataUsage::Intermediate);
            IE_ASSERT(cmxData->parentDataEdge() == nullptr);

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

            env.log->debug("CMX data [%s] consumer [%s]", cmxData->name(), cmxConsumer->name());
            VPU_LOGGER_SECTION(env.log);

            Data ddrCopy;
            if (cmxData->attrs().has("ddrCopy")) {
                env.log->debug("It already has DDR spill");

                ddrCopy = cmxData->attrs().get<Data>("ddrCopy");
            } else {
                env.log->debug("Create new DDR spill");

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
                    ddrCopy);
                copyStage->attrs().set<bool>("CMX-to-DDR", true);

                cmxData->attrs().set("ddrCopy", ddrCopy);
            }

            model->replaceStageInput(cmxConsumerEdge, ddrCopy);

            env.log->debug("Update child datas");
            for (const auto& childDataEdge : cmxData->childDataEdges()) {
                VPU_LOGGER_SECTION(env.log);

                auto order = childDataEdge->order();

                if (order == SharedDataOrder::ParentWritesToChild &&
                    childDataEdge->connection() == cmxConsumer) {
                    auto childData = childDataEdge->child();

                    auto mode = childDataEdge->mode();
                    auto offset = childDataEdge->attrs().getOrDefault<DimValues>("offset", DimValues());

                    env.log->debug("Child data [%s] : mode [%v] offset [%v]", childData->name(), mode, offset);

                    model->connectDatas()
                        .parent(ddrCopy)
                        .child(childData)
                        .mode(mode)
                        .order(order)
                        .offset(offset)
                        .done();

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

void PassImpl::copyHwMisalignedInput(const Model::Ptr& model) {
    VPU_PROFILE(copyHwMisalignedInput);

    const auto& env = CompileEnv::get();

    env.log->debug("Add Copy for misaligned HW inputs");
    VPU_LOGGER_SECTION(env.log);

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW) {
            continue;
        }

        auto input = stage->input(0);
        IE_ASSERT(input->location() != DataLocation::None);

        if (input->memoryOffset() % 16 != 0) {
            env.log->debug("HW Stage [%s] input [%s]", stage->name(), input->name());

            auto newInput = model->duplicateData(
                input,
                "@aligned-ptr");
            newInput->setMemReqs(MemoryType::DDR);

            _stageBuilder->addCopyStage(
                model,
                stage->name() + "@align-input-ptr",
                stage->origLayer(),
                input,
                newInput);

            model->replaceStageInput(stage->inputEdge(0), newInput);
        }
    }
}

//
// Try to put HW inputs to CMX if possible
//

void PassImpl::packDataInCmx(const Model::Ptr& model) {
    VPU_PROFILE(packDataInCmx);

    const auto& env = CompileEnv::get();

    env.log->debug("Try to use CMX for HW inputs");
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

        env.log->debug("Try use CMX for Data [%s]", curCandidate->name());
        VPU_LOGGER_SECTION(env.log);

        IE_ASSERT(curCandidate->parentDataEdge() == nullptr);
        IE_ASSERT(curCandidate->usage() == DataUsage::Intermediate);

        auto curMemoryType = curCandidate->memReqs();
        IE_ASSERT(curMemoryType == MemoryType::DDR);

        loopOverData(curCandidate, [](const Data& subData) {
            subData->setMemReqs(MemoryType::CMX);
            return DataLoopStatus::NextChild;
        });

        auto allocRes = runAllocator(model, true);
        env.log->debug("Allocation result : %v", allocRes.status);

        if (allocRes.status != AllocationStatus::OK) {
            env.log->debug("Revert CMX usage for Data [%s]", curCandidate->name());

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
