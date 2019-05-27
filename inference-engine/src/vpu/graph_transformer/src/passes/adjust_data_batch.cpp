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

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(adjustDataBatch);

    for (const auto& stage : model->getStages()) {
        //
        // Get stage information
        //

        auto stageInfo = stage->getBatchSupportInfo();

        if (stageInfo.empty()) {
            continue;
        }

        //
        // Get batch size
        //

        int batchSize = -1;

        for (const auto& input : stage->inputs()) {
            auto it = stageInfo.find(input);
            if (it == stageInfo.end()) {
                continue;
            }

            auto curReq = it->second;

            if (curReq == BatchSupport::Split) {
                if (batchSize < 0) {
                    batchSize = input->desc().dim(Dim::N, 1);
                } else {
                    IE_ASSERT(batchSize == input->desc().dim(Dim::N, 1));
                }
            }
        }

        IE_ASSERT(batchSize > 0);

        for (const auto& output : stage->outputs()) {
            IE_ASSERT(stageInfo.at(output) == BatchSupport::Split);
            IE_ASSERT(batchSize == output->desc().dim(Dim::N, 1));
        }

        if (batchSize == 1) {
            continue;
        }

        //
        // Create tiles and replicate input
        //

        DataMap<DataVector> inputTiles;
        DataMap<DataVector> outputTiles;

        for (const auto& input : stage->inputs()) {
            auto it = stageInfo.find(input);
            if (it == stageInfo.end()) {
                continue;
            }

            auto curReq = it->second;
            if (curReq == BatchSupport::Split) {
                auto newDesc = input->desc();
                newDesc.setDim(Dim::N, 1);

                inputTiles[input].reserve(batchSize);
                for (int batchInd = 0; batchInd < batchSize; ++batchInd) {
                    auto postfix = formatString("@batch=%d/%d", batchInd + 1, batchSize);

                    auto inputTile = model->duplicateData(
                        input,
                        postfix,
                        newDesc);

                    inputTiles[input].emplace_back(std::move(inputTile));
                }
            } else if (curReq == BatchSupport::ReplicateConstContent) {
                IE_ASSERT(input->usage() == DataUsage::Const);

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
        }
        for (const auto& output : stage->outputs()) {
            auto newDesc = output->desc();
            newDesc.setDim(Dim::N, 1);

            outputTiles[output].reserve(batchSize);
            for (int batchInd = 0; batchInd < batchSize; ++batchInd) {
                auto postfix = formatString("@batch=%d/%d", batchInd + 1, batchSize);

                auto outputTile = model->duplicateData(
                    output,
                    postfix,
                    newDesc);

                outputTiles[output].emplace_back(std::move(outputTile));
            }
        }

        //
        // Replicate stage
        //

        for (int batchInd = 0; batchInd < batchSize; ++batchInd) {
            auto postfix = formatString("@batch=%d/%d", batchInd + 1, batchSize);

            DataVector newInputs;
            for (const auto& inEdge : stage->inputEdges()) {
                if (stageInfo.count(inEdge->input()) == 0) {
                    newInputs.emplace_back(inEdge->input());
                    continue;
                }

                auto curReq = stageInfo[inEdge->input()];

                if (curReq == BatchSupport::Split) {
                    newInputs.emplace_back(inputTiles.at(inEdge->input())[batchInd]);
                } else if (curReq == BatchSupport::ReplicateConstContent) {
                    const auto& replicatedDatas = inEdge->input()->attrs().get<ReplicatedDataMap>("replicatedDatas");
                    newInputs.emplace_back(replicatedDatas.at(batchSize));
                }
            }

            DataVector newOutputs;
            for (const auto& output : stage->outputs()) {
                newOutputs.emplace_back(outputTiles.at(output)[batchInd]);
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

        //
        // Create split/concat stages
        //

        model->disconnectStageDatas(stage);

        for (const auto& p : inputTiles) {
            _stageBuilder->addSplitStage(
                model,
                stage->name() + "@split-batch",
                stage->origLayer(),
                Dim::N,
                p.first,
                p.second);
        }

        for (const auto& p : outputTiles) {
            _stageBuilder->addConcatStage(
                model,
                stage->name() + "@concat-batch",
                stage->origLayer(),
                Dim::N,
                p.second,
                p.first);
        }

        //
        // Remove original stage
        //

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::adjustDataBatch() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
