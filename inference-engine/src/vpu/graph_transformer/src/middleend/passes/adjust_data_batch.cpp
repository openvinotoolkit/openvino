// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/stages/iteration_rule.hpp"
#include "vpu/middleend/pass_manager.hpp"
#include "vpu/model/data_contents/replicated_data_content.hpp"

#include <utility>
#include <string>
#include <set>
#include <unordered_map>
#include <memory>

namespace vpu {

namespace {

using ReplicatedDataMap = std::unordered_map<int, Data>;
using BatchTilesMap = DataMap<DataVector>;

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override;

private:
    StageList collectAllStageToSplit(const Model& model);

    StageList extractNextSubGraph(StageList& stagesToSplit);

    void duplicate(const Model& model, const StageList& subgraph);
    void wrapInLoop(const Model& model, const StageList& subgraph);

    void processStageInputs(
        const Stage& stage,
        const Model& model,
        const StageList& curSubGraph,
        DataMap<DataVector>& subGraphInputTiles,
        BatchTilesMap& batchTilesMap);

    void splitStageInput(
        const StageInput& inEdge,
        const Model& model,
        const StageList& curSubGraph,
        DataMap<DataVector>& subGraphInputTiles,
        BatchTilesMap& batchTilesMap);

    void processStageOutputs(
        const Stage& stage,
        const Model& model,
        const StageList& curSubGraph,
        DataMap<DataVector>& subGraphOutputTiles,
        BatchTilesMap& batchTilesMap);

    void replicateStage(
        const Stage& stage,
        const Model& model,
        const BatchTilesMap& batchTilesMap);

    void removeOriginalStages(
        const StageList& curSubGraph,
        const Model& model);

    void addSplitConcatPair(
        const DataMap<DataVector>& subGraphInputTiles,
        const DataMap<DataVector>& subGraphOutputTiles,
        const Model& model);

private:
    StageBuilder::Ptr _stageBuilder;

    // Without index LoopStart/LoopEnd pairs will have the same name for different batched sub-graphs. Since
    // performance map from GetPerformanceCount contains unique keys only, in performance report
    // such sub-graphs reported incorrectly.
    std::size_t loopIndex = 0;

    // Using Loop construction to process sub-graph with batch may produce extra copy operations
    // in comparison with duplicating sub-graph body + Split/Concat pair.
    // It negatively affects performance of some topologies (e.g. ctpn).
    // To overcome this issue heuristic based on sub-graph's size has been introduced:
    // if sub-graph small enough we try not to use Loop in order to avoid unnecessary copy stages
    // or inject them, otherwise sub-graph is wrapped in a Loop.
    static constexpr std::size_t s_subgraphSizeThreshold = 2;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(adjustDataBatch);

    auto stagesToSplit = collectAllStageToSplit(model);

    while (!stagesToSplit.empty()) {
        auto curSubGraph = extractNextSubGraph(stagesToSplit);
        IE_ASSERT(!curSubGraph.empty());

        if (curSubGraph.size() <= s_subgraphSizeThreshold) {
            duplicate(model, curSubGraph);
        } else {
            wrapInLoop(model, curSubGraph);
        }
    }
}

void PassImpl::duplicate(const Model& model, const StageList& subgraph) {
    DataMap<DataVector> subGraphInputTiles;
    DataMap<DataVector> subGraphOutputTiles;
    BatchTilesMap batchTilesMap;

    for (const auto& stage : subgraph) {
        processStageInputs(stage, model, subgraph, subGraphInputTiles, batchTilesMap);
        processStageOutputs(stage, model, subgraph, subGraphOutputTiles, batchTilesMap);
        replicateStage(stage, model, batchTilesMap);
    }

    removeOriginalStages(subgraph, model);

    addSplitConcatPair(subGraphInputTiles, subGraphOutputTiles, model);
}

void PassImpl::wrapInLoop(const Model& model, const StageList& subgraph) {
    const auto isInputData = [](const StageList& subgraph, const Data& data) {
        return !data->producer() || !subgraph.has(data->producer());
    };

    const auto isOutputData = [](const StageList& subgraph, const Data& data) {
        for (const auto& consumer : data->consumers()) {
            if (!subgraph.has(consumer)) {
                return true;
            }
        }
        return data->usage() == DataUsage::Output;
    };

    DataVector loopStartInputs, loopStartOutputs, loopEndInputs, loopEndOutputs;
    IterationComponents startIterationComponents, endIterationComponents;

    for (const auto& stage : subgraph) {
        const auto& batchInfo = stage->getBatchSupportInfo();
        for (const auto& inputEdge : stage->inputEdges()) {
            const auto& input = inputEdge->input();
            if (isInputData(subgraph, input)) {
                const bool withBatch = batchInfo.hasInput(inputEdge);
                // if producer exists, data object should be passed to LoopStart in any case
                // to be sure producer will be executed before loop starts
                const bool hasProducer = input->producer() != nullptr;
                if (withBatch || hasProducer) {
                    loopStartInputs.push_back(input);

                    // data object needs to be kept alive during whole loop execution
                    // otherwise some stage may overwrite this input and on the next iteration stage will get corrupted data
                    // to do so mark this data object as LoopEnd input
                    loopEndInputs.push_back(input);

                    // LoopStart stage requires corresponding between inputs and outputs data object
                    // to propagate data order requirements
                    auto outputDescriptor = input->desc();
                    if (withBatch) {
                        outputDescriptor.setDim(Dim::N, 1);
                    }
                    const auto loopStartOutput = model->duplicateData(
                        input,
                        formatString("@LoopStartOutput@%v", input->name()),
                        outputDescriptor);
                    loopStartOutputs.push_back(loopStartOutput);

                    if (withBatch) {
                        const auto rule = IterationRule{Dim::N, 0, 1, -1};
                        startIterationComponents.emplace(std::make_pair(loopStartInputs.size() - 1, rule), loopStartOutputs.size() - 1);
                    } else {
                        // do not allocate extra memory since there cannot be back-edge connection
                        input->attrs().set<Data>("start-shared-allocation", loopStartOutput);
                    }

                    model->replaceStageInput(inputEdge, loopStartOutput);
                }
            }
        }

        for (const auto& outputEdge : stage->outputEdges()) {
            const auto originalOutput = outputEdge->output();
            auto descriptor = originalOutput->desc();
            descriptor.setDim(Dim::N, 1);

            const auto output = model->duplicateData(
                originalOutput,
                formatString("@batch@%v", originalOutput->name()),
                descriptor);
            model->replaceStageOutput(outputEdge, output);

            if (isOutputData(subgraph, originalOutput)) {
                loopEndInputs.push_back(output);
                loopEndOutputs.push_back(originalOutput);
                const auto rule = IterationRule{Dim::N, 0, 1, -1};
                endIterationComponents.emplace(std::make_pair(loopEndOutputs.size() - 1, rule), loopEndInputs.size() - 1);
            } else {
                for (const auto& consumerEdge : originalOutput->consumerEdges()) {
                    model->replaceStageInput(consumerEdge, output);
                }
            }
        }
    }

    auto loopStart = _stageBuilder->addLoopStartStage(model, formatString("LoopStart@Batch@{}", loopIndex), loopStartInputs, loopStartOutputs);
    auto loopEnd = _stageBuilder->addLoopEndStage(model, formatString("LoopEnd@Batch@{}", loopIndex), loopEndInputs, loopEndOutputs);
    ++loopIndex;

    loopStart->attrs().set("start-iteration-components", startIterationComponents);
    loopEnd->attrs().set("end-iteration-components", endIterationComponents);
    loopStart->attrs().set("loop-end", loopEnd);
    loopStart->attrs().set<uint32_t>("iterations-count", subgraph.front()->attrs().get<int>("batchSize"));
    loopEnd->attrs().set<uint32_t>("iterations-count", subgraph.front()->attrs().get<int>("batchSize"));
}

//
// Collect all stages that doesn't support batch
//

StageList PassImpl::collectAllStageToSplit(const Model& model) {
    StageList stagesToSplit(&StageNode::posForPassList);

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
        stagesToSplit.push_back(stage);
    }

    return stagesToSplit;
}

//
// Extract next sub-graph for process, it should be completely independent from other Stages
//

StageList PassImpl::extractNextSubGraph(StageList& stagesToSplit) {
    //
    // Add new Stage to the sub-graph only if it depends on Stages from sub-graph only
    //

    StageList subGraph(&StageNode::posForPassList);

    for (const auto& stage : stagesToSplit) {
        bool isInternalStage = true;
        for (const auto& prevStage : stage->prevStages()) {
            if (!subGraph.has(prevStage)) {
                isInternalStage = false;
                break;
            }
        }
        if (isInternalStage || subGraph.empty()) {
            stagesToSplit.erase(stage);
            subGraph.push_back(stage);
        }

        bool shouldStop = false;
        bool several_consumers = false;
        for (const auto& outputEdge : stage->outputEdges()) {
            const auto originalOutput = outputEdge->output();
            if (originalOutput->numConsumers() > 1) {
                several_consumers = true;
                break;
            }
        }
        for (const auto& nextStage : stage->nextStages()) {
        if (!stagesToSplit.has(nextStage) && several_consumers) {
                shouldStop = true;
                break;
            }
        }
        if (shouldStop) {
            break;
        }
    }

    return subGraph;
}

void PassImpl::processStageInputs(
    const Stage& stage,
    const Model& model,
    const StageList& curSubGraph,
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
        }
    }
}

void PassImpl::splitStageInput(
    const StageInput& inEdge,
    const Model& model,
    const StageList& curSubGraph,
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
        if (curSubGraph.has(producer)) {
            isInternalInput = true;
        }
    }
    if (!isInternalInput) {
        auto res = subGraphInputTiles.emplace(input, batchTiles);
        IE_ASSERT(res.second);
    }
}

void PassImpl::processStageOutputs(
    const Stage& stage,
    const Model& model,
    const StageList& curSubGraph,
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
            if (!curSubGraph.has(consumer)) {
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
    const Model& model,
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
            }
        }

        DataVector newOutputs;
        for (const auto& output : stage->outputs()) {
            const auto& batchTiles = batchTilesMap.at(output);
            IE_ASSERT(batchTiles.size() == batchSize);

            newOutputs.emplace_back(batchTiles[batchInd]);
        }

        auto tileStage = model->duplicateStage(
            stage,
            postfix,
            newInputs,
            newOutputs);

        tileStage->attrs().set<int>("batchInd", batchInd);

        if (stage->type() == StageType::StubConv) {
            tileStage->attrs().set("origConvOutput", newOutputs[0]->desc());
        }
    }
}

void PassImpl::removeOriginalStages(
    const StageList& curSubGraph,
    const Model& model) {
    for (const auto& stage : curSubGraph) {
        model->removeStage(stage);
    }
}

void PassImpl::addSplitConcatPair(
    const DataMap<DataVector>& subGraphInputTiles,
    const DataMap<DataVector>& subGraphOutputTiles,
    const Model& model) {
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
