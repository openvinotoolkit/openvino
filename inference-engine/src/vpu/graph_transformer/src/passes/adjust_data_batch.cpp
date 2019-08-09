// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <cmath>
#include <list>
#include <set>
#include <unordered_map>
#include <memory>

namespace vpu {

namespace {

using ReplicatedDataMap = std::unordered_map<int, Data>;
using StagesOrderedSet = std::set<Stage, StageNode::StageIndexCmp>;
using BatchTilesMap = DataMap<DataVector>;

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StagesOrderedSet collectAllStageToSplit(const Model::Ptr& model);

    StagesOrderedSet extractNextSubGraph(StagesOrderedSet& stagesToSplit);

    void processStageInputs(
            const Stage& stage,
            const Model::Ptr& model,
            const StagesOrderedSet& curSubGraph,
            DataMap<DataVector>& subGraphInputTiles,
            BatchTilesMap& batchTilesMap);
    void splitStageInput(
            const StageInput& inEdge,
            const Model::Ptr& model,
            const StagesOrderedSet& curSubGraph,
            DataMap<DataVector>& subGraphInputTiles,
            BatchTilesMap& batchTilesMap);
    void replicateStageInput(
            const StageInput& inEdge,
            const Model::Ptr& model);

    void processStageOutputs(
            const Stage& stage,
            const Model::Ptr& model,
            const StagesOrderedSet& curSubGraph,
            DataMap<DataVector>& subGraphOutputTiles,
            BatchTilesMap& batchTilesMap);

    void replicateStage(
            const Stage& stage,
            const Model::Ptr& model,
            const BatchTilesMap& batchTilesMap);

    void removeOriginalStages(
            const StagesOrderedSet& curSubGraph,
            const Model::Ptr& model);

    void addSplitConcatPair(
            const DataMap<DataVector>& subGraphInputTiles,
            const DataMap<DataVector>& subGraphOutputTiles,
            const Model::Ptr& model);

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(adjustDataBatch);

    auto stagesToSplit = collectAllStageToSplit(model);

    while (!stagesToSplit.empty()) {
        auto curSubGraph = extractNextSubGraph(stagesToSplit);
        IE_ASSERT(!curSubGraph.empty());

        DataMap<DataVector> subGraphInputTiles;
        DataMap<DataVector> subGraphOutputTiles;
        BatchTilesMap batchTilesMap;

        for (const auto& stage : curSubGraph) {
            processStageInputs(stage, model, curSubGraph, subGraphInputTiles, batchTilesMap);
            processStageOutputs(stage, model, curSubGraph, subGraphOutputTiles, batchTilesMap);
            replicateStage(stage, model, batchTilesMap);
        }

        removeOriginalStages(curSubGraph, model);

        addSplitConcatPair(subGraphInputTiles, subGraphOutputTiles, model);
    }
}

//
// Collect all stages that doesn't support batch
//

StagesOrderedSet PassImpl::collectAllStageToSplit(const Model::Ptr& model) {
    StagesOrderedSet stagesToSplit;

    for (const auto& stage : model->getStages()) {
        //
        // Get stage information
        //

        const auto& stageInfo = stage->getBatchSupportInfo();

        if (stageInfo.empty()) {
            continue;
        }

        //
        // Get batch size
        //

        int batchSize = -1;

        for (const auto& inEdge : stage->inputEdges()) {
            if (!stageInfo.hasInput(inEdge)) {
                continue;
            }

            auto curReq = stageInfo.getInput(inEdge);

            if (curReq == BatchSupport::Split) {
                if (batchSize < 0) {
                    batchSize = inEdge->input()->desc().dim(Dim::N, 1);
                } else {
                    IE_ASSERT(batchSize == inEdge->input()->desc().dim(Dim::N, 1));
                }
            }
        }

        IE_ASSERT(batchSize > 0);

        for (const auto& outEdge : stage->outputEdges()) {
            IE_ASSERT(stageInfo.getOutput(outEdge) == BatchSupport::Split);
            IE_ASSERT(batchSize == outEdge->output()->desc().dim(Dim::N, 1));
        }

        if (batchSize == 1) {
            continue;
        }

        stage->attrs().set("batchSize", batchSize);
        stagesToSplit.emplace(stage);
    }

    return stagesToSplit;
}

//
// Extract next sub-graph for process, it should be completely independent from other Stages
//

StagesOrderedSet PassImpl::extractNextSubGraph(StagesOrderedSet& stagesToSplit) {
    //
    // Add new Stage to the sub-graph only if it depends on Stages from sub-graph only
    //

    StagesOrderedSet subGraph;

    for (const auto& stage : stagesToSplit) {
        bool isInternalStage = true;
        for (const auto& prevStage : stage->prevStages()) {
            if (subGraph.count(prevStage) == 0) {
                isInternalStage = false;
                break;
            }
        }
        if (isInternalStage || subGraph.empty()) {
            subGraph.emplace(stage);
        }

        bool shouldStop = false;
        for (const auto& nextStage : stage->nextStages()) {
            if (stagesToSplit.count(nextStage) == 0) {
                shouldStop = true;
                break;
            }
        }
        if (shouldStop) {
            break;
        }
    }

    for (const auto& stage : subGraph) {
        stagesToSplit.erase(stage);
    }

    return subGraph;
}

void PassImpl::processStageInputs(
        const Stage& stage,
        const Model::Ptr& model,
        const StagesOrderedSet& curSubGraph,
        DataMap<DataVector>& subGraphInputTiles,
        BatchTilesMap& batchTilesMap) {
    const auto& stageInfo = stage->getBatchSupportInfo();

    for (const auto& inEdge : stage->inputEdges()) {
        if (!stageInfo.hasInput(inEdge)) {
            continue;
        }

        auto curReq = stageInfo.getInput(inEdge);

        if (curReq == BatchSupport::Split) {
            splitStageInput(inEdge, model, curSubGraph, subGraphInputTiles, batchTilesMap);
        } else if (curReq == BatchSupport::ReplicateConstContent) {
            replicateStageInput(inEdge, model);
        }
    }
}

void PassImpl::splitStageInput(
        const StageInput& inEdge,
        const Model::Ptr& model,
        const StagesOrderedSet& curSubGraph,
        DataMap<DataVector>& subGraphInputTiles,
        BatchTilesMap& batchTilesMap) {
    const auto& input = inEdge->input();
    const auto& stage = inEdge->consumer();

    auto batchSize = stage->attrs().get<int>("batchSize");

    auto newDesc = input->desc();
    newDesc.setDim(Dim::N, 1);

    auto& batchTiles = batchTilesMap[input];
    if (!batchTiles.empty()) {
        IE_ASSERT(batchTiles.size() == batchSize);
        return;
    }

    batchTiles.resize(batchSize);
    for (int batchInd = 0; batchInd < batchSize; ++batchInd) {
        auto postfix = formatString("@batch=%d/%d", batchInd + 1, batchSize);

        batchTiles[batchInd] = model->duplicateData(
            input,
            postfix,
            newDesc);
    }

    bool isInternalInput = false;
    if (auto producer = input->producer()) {
        if (curSubGraph.count(producer) != 0) {
            isInternalInput = true;
        }
    }
    if (!isInternalInput) {
        auto res = subGraphInputTiles.emplace(input, batchTiles);
        IE_ASSERT(res.second);
    }
}

void PassImpl::replicateStageInput(
        const StageInput& inEdge,
        const Model::Ptr& model) {
    const auto& input = inEdge->input();
    const auto& stage = inEdge->consumer();

    IE_ASSERT(input->usage() == DataUsage::Const);
    auto batchSize = stage->attrs().get<int>("batchSize");

    auto& replicatedDatas = input->attrs().getOrSet<ReplicatedDataMap>("replicatedDatas", ReplicatedDataMap());
    if (replicatedDatas.count(batchSize) == 0) {
        auto content = input->content();
        IE_ASSERT(content != nullptr);

        auto perm = input->desc().dimsOrder().toPermutation();
        auto dims = input->desc().dims();

        int maxDimDigit = -1;
        for (auto d : perm) {
            maxDimDigit = std::max(maxDimDigit, static_cast<int>(d));
        }
        IE_ASSERT(maxDimDigit >= 0);

        perm.emplace_back(static_cast<Dim>(maxDimDigit + 1));
        dims.set(perm.back(), batchSize);

        DataDesc newDesc(input->desc().type(), DimsOrder::fromPermutation(perm), dims);

        replicatedDatas[batchSize] = model->duplicateData(
            input,
            formatString("@replicated=%d", batchSize),
            newDesc,
            replicateContent(content, batchSize));
    }
}

void PassImpl::processStageOutputs(
        const Stage& stage,
        const Model::Ptr& model,
        const StagesOrderedSet& curSubGraph,
        DataMap<DataVector>& subGraphOutputTiles,
        BatchTilesMap& batchTilesMap) {
    auto batchSize = stage->attrs().get<int>("batchSize");

    for (const auto& output : stage->outputs()) {
        auto newDesc = output->desc();
        newDesc.setDim(Dim::N, 1);

        auto& batchTiles = batchTilesMap[output];
        if (!batchTiles.empty()) {
            IE_ASSERT(batchTiles.size() == batchSize);
            continue;
        }

        batchTiles.resize(batchSize);
        for (int batchInd = 0; batchInd < batchSize; ++batchInd) {
            auto postfix = formatString("@batch=%d/%d", batchInd + 1, batchSize);

            batchTiles[batchInd] = model->duplicateData(
                output,
                postfix,
                newDesc);
        }

        bool isInternalOutput = output->usage() == DataUsage::Intermediate;
        for (const auto& consumer : output->consumers()) {
            if (curSubGraph.count(consumer) == 0) {
                isInternalOutput = false;
                break;
            }
        }
        if (!isInternalOutput) {
            auto res = subGraphOutputTiles.emplace(output, batchTiles);
            IE_ASSERT(res.second);
        }
    }
}

void PassImpl::replicateStage(
        const Stage& stage,
        const Model::Ptr& model,
        const BatchTilesMap& batchTilesMap) {
    const auto& stageInfo = stage->getBatchSupportInfo();
    auto batchSize = stage->attrs().get<int>("batchSize");

    for (int batchInd = 0; batchInd < batchSize; ++batchInd) {
        auto postfix = formatString("@batch=%d/%d", batchInd + 1, batchSize);

        DataVector newInputs;
        for (const auto& inEdge : stage->inputEdges()) {
            if (!stageInfo.hasInput(inEdge)) {
                newInputs.emplace_back(inEdge->input());
                continue;
            }

            auto curReq = stageInfo.getInput(inEdge);

            if (curReq == BatchSupport::Split) {
                const auto& batchTiles = batchTilesMap.at(inEdge->input());
                IE_ASSERT(batchTiles.size() == batchSize);

                newInputs.emplace_back(batchTiles[batchInd]);
            } else if (curReq == BatchSupport::ReplicateConstContent) {
                const auto& replicatedDatas = inEdge->input()->attrs().get<ReplicatedDataMap>("replicatedDatas");
                newInputs.emplace_back(replicatedDatas.at(batchSize));
            }
        }

        DataVector newOutputs;
        for (const auto& output : stage->outputs()) {
            const auto& batchTiles = batchTilesMap.at(output);
            IE_ASSERT(batchTiles.size() == batchSize);

            newOutputs.emplace_back(batchTiles[batchInd]);
        }

        auto tileStage = model->duplicateStage(
            stage->name() + postfix,
            stage,
            newInputs,
            newOutputs);

        tileStage->attrs().set<int>("batchInd", batchInd);

        if (stage->type() == StageType::StubConv) {
            tileStage->attrs().set("origConvOutput", newOutputs[0]->desc());
        }
    }
}

void PassImpl::removeOriginalStages(
        const StagesOrderedSet& curSubGraph,
        const Model::Ptr& model) {
    for (const auto& stage : curSubGraph) {
        model->removeStage(stage);
    }
}

void PassImpl::addSplitConcatPair(
        const DataMap<DataVector>& subGraphInputTiles,
        const DataMap<DataVector>& subGraphOutputTiles,
        const Model::Ptr& model) {
    for (const auto& p : subGraphInputTiles) {
        _stageBuilder->addSplitStage(
            model,
            p.first->name() + "@split-batch",
            nullptr,
            Dim::N,
            p.first,
            p.second);
    }

    for (const auto& p : subGraphOutputTiles) {
        if (p.first->usage() == DataUsage::Intermediate) {
            IE_ASSERT(p.first->numConsumers() > 0);
        }

        _stageBuilder->addConcatStage(
            model,
            p.first->name() + "@concat-batch",
            nullptr,
            Dim::N,
            p.second,
            p.first);
    }
}

}  // namespace

Pass::Ptr PassManager::adjustDataBatch() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
