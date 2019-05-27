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

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW) {
            continue;
        }

        auto output = stage->output(0);

        if (output->usage() == DataUsage::Output) {
            auto newOutput = model->duplicateData(
                output,
                "@intermediate");

            model->replaceStageOutput(stage->outputEdge(0), newOutput);

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
            setDataMemReqs(hwInput, MemoryType::CMX);
        } else {
            setDataMemReqs(hwOutput, MemoryType::CMX);
        }
    }
}

//
// Allocate Const/Input/Output datas
//

void PassImpl::allocNonIntermediateData(const Model::Ptr& model) {
    VPU_PROFILE(allocNonIntermediateData);

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

    auto& allocator = model->getAllocator();

    for (;;) {
        auto allocRes = runAllocator(model);
        if (allocRes.status == AllocationStatus::OK)
            break;

        auto failedStage = allocRes.failedStage;
        IE_ASSERT(failedStage != nullptr);

        auto failedStageInd = failedStage->index();
        IE_ASSERT(failedStageInd >= 0);

        //
        // Try to flush Data allocated in CMX
        //

        auto allCmxDatas = allocator.getAllocatedDatas(MemoryType::CMX);

        if (allCmxDatas.empty()) {
            if (allocRes.status == AllocationStatus::SHAVES_FAILED) {
                VPU_THROW_EXCEPTION
                    << "Can't allocate SHAVEs for stage " << failedStage->name();
            } else {
                VPU_THROW_EXCEPTION
                    << "Can't satisfy data location requirements for stage " << failedStage->name();
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

        for (const auto& cmxConsumerEdge : cmxConsumerEdges) {
            auto cmxData = cmxConsumerEdge->input();
            auto cmxConsumer = cmxConsumerEdge->consumer();

            auto& ddrCopies = cmxData->attrs().getOrSet<DataVector>("ddrCopies", DataVector());
            ddrCopies.reserve(1);

            const auto& strideReqsInfo = cmxConsumer->getDataStridesRequirements();

            Data ddrCopy;
            for (const auto& ddrCandidate : ddrCopies) {
                if (strideReqsInfo.count(cmxConsumerEdge->input()) != 0) {
                    const auto& strideReqs = strideReqsInfo.at(cmxConsumerEdge->input());

                    if (!ddrCandidate->checkStrides(strideReqs)) {
                        continue;
                    }
                }

                ddrCopy = ddrCandidate;
                break;
            }

            if (ddrCopy == nullptr) {
                ddrCopy = model->duplicateData(
                    cmxData,
                    "@DDR-copy");

                if (strideReqsInfo.count(cmxConsumerEdge->input()) != 0) {
                    const auto& strideReqs = strideReqsInfo.at(cmxConsumerEdge->input());
                    ddrCopy->updateRequiredStrides(strideReqs);
                }
                ddrCopy->setMemReqs(MemoryType::DDR);

                auto copyStage = _stageBuilder->addCopyStage(
                    model,
                    formatString("%s@move-to-DDR", cmxData->name()),
                    failedStage->origLayer(),
                    cmxData,
                    ddrCopy);

                copyStage->attrs().set<bool>("CMX-to-DDR", true);

                ddrCopies.emplace_back(ddrCopy);
            }

            model->replaceStageInput(cmxConsumerEdge, ddrCopy);

            for (const auto& childDataEdge : cmxData->childDataEdges()) {
                auto order = childDataEdge->order();

                if (order == SharedDataOrder::ParentWritesToChild &&
                    childDataEdge->connection() == cmxConsumer) {
                    auto childData = childDataEdge->child();

                    model->connectDatas()
                        .parent(ddrCopy)
                        .child(childData)
                        .mode(childDataEdge->mode())
                        .order(order)
                        .offset(childDataEdge->attrs().getOrDefault<DimValues>("offset", DimValues()))
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

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW) {
            continue;
        }

        auto input = stage->input(0);
        IE_ASSERT(input->location() != DataLocation::None);

        if (input->memoryOffset() % 16 != 0) {
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

        IE_ASSERT(curCandidate->parentDataEdge() == nullptr);
        IE_ASSERT(curCandidate->usage() == DataUsage::Intermediate);

        auto curMemoryType = curCandidate->memReqs();
        IE_ASSERT(curMemoryType == MemoryType::DDR);

        loopOverData(curCandidate, [](const Data& subData) {
            subData->setMemReqs(MemoryType::CMX);
            return DataLoopStatus::NextChild;
        });

        auto allocRes = runAllocator(model, true);
        if (allocRes.status != AllocationStatus::OK) {
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
