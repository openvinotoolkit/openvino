// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <memory>
#include <array>
#include <string>
#include <list>
#include <unordered_set>
#include <vector>
#include <set>
#include <tuple>
#include <limits>

#include <precision_utils.h>

#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/utility.hpp>

namespace vpu {

namespace {

std::tuple<Data, Data, Data> createWeightableInputsForDepthConv(
        const Model& model,
        const Data& origWeights,
        const Data& origBiases,
        const Data& origScales,
        const Stage& origStage,
        int tileSize, int tileOffset,
        const std::string& postfix) {
    auto kernelSizeX = origStage->attrs().get<int>("kernelSizeX");
    auto kernelSizeY = origStage->attrs().get<int>("kernelSizeY");

    Data newWeights;
    {
        auto content = origWeights->content();
        IE_ASSERT(content != nullptr);

        auto origWeightsVals = content->get<fp16_t>();
        IE_ASSERT(origWeightsVals != nullptr);

        size_t newWeightsSize = kernelSizeX * kernelSizeY * tileSize * tileSize;

        auto newWeightsBlob = ie::make_shared_blob<fp16_t>(ie::TensorDesc(
            ie::Precision::FP16,
            {newWeightsSize},
            ie::Layout::C));
        newWeightsBlob->allocate();

        auto inPtr = origWeightsVals + kernelSizeX * kernelSizeY * tileOffset;
        auto outPtr = newWeightsBlob->buffer().as<fp16_t*>();

        std::fill_n(outPtr, newWeightsSize, ie::PrecisionUtils::f32tof16(0.0f));

        for (int idx = 0; idx < tileSize; ++idx) {
            auto srcSlicePtr = inPtr + idx * kernelSizeX * kernelSizeY;
            auto dstSlicePtr = outPtr + idx * (kernelSizeX * kernelSizeY) * (tileSize + 1);
            std::copy_n(srcSlicePtr, kernelSizeX * kernelSizeY, dstSlicePtr);
        }

        newWeights = model->duplicateData(
            origWeights,
            postfix,
            DataDesc({kernelSizeX, kernelSizeY, tileSize, tileSize}),
            ieBlobContent(newWeightsBlob));
    }

    auto newBiases = origBiases;
    if (origBiases->usage() != DataUsage::Fake) {
        auto content = origBiases->content();
        IE_ASSERT(content != nullptr);

        auto origBiasesVals = content->get<fp16_t>();
        IE_ASSERT(origBiasesVals != nullptr);

        auto newBiasesBlob = ie::make_shared_blob<fp16_t>(ie::TensorDesc(
            ie::Precision::FP16,
            {static_cast<size_t>(tileSize)},
            ie::Layout::C));
        newBiasesBlob->allocate();

        auto inPtr = origBiasesVals + tileOffset;
        auto outPtr = newBiasesBlob->buffer().as<fp16_t*>();

        std::copy_n(inPtr, tileSize, outPtr);

        newBiases = model->duplicateData(
            origBiases,
            postfix,
            DataDesc({tileSize}),
            ieBlobContent(newBiasesBlob));
    }

    auto newScales = origScales;
    if (origScales->usage() != DataUsage::Fake) {
        auto content = origScales->content();
        IE_ASSERT(content != nullptr);

        auto origScalesVals = content->get<fp16_t>();
        IE_ASSERT(origScalesVals != nullptr);

        auto newScalesBlob = ie::make_shared_blob<fp16_t>(ie::TensorDesc(
            ie::Precision::FP16,
            {static_cast<size_t>(tileSize)},
            ie::Layout::C));
        newScalesBlob->allocate();

        auto inPtr = origScalesVals + tileOffset;
        auto outPtr = newScalesBlob->buffer().as<fp16_t*>();

        std::copy_n(inPtr, tileSize, outPtr);

        newScales = model->duplicateData(
            origScales,
            postfix,
            DataDesc({tileSize}),
            ieBlobContent(newScalesBlob));
    }

    return std::make_tuple(newWeights, newBiases, newScales);
}

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(splitHwDepthConv);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubConv) {
            continue;
        }

        auto tryHW = stage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto scales = stage->input(3);
        auto output = stage->output(0);

        if (input->desc().dim(Dim::C) != output->desc().dim(Dim::C)) {
            continue;
        }

        auto groupSize = stage->attrs().get<int>("groupSize");
        if (groupSize != input->desc().dim(Dim::C)) {
            continue;
        }

        //
        // Collect cost per tile
        //

        using OptTile = std::tuple<int, double>;
        std::vector<OptTile> optTiles;
        optTiles.reserve(output->desc().dim(Dim::C));

        for (int curTileSize = 1; curTileSize < output->desc().dim(Dim::C); curTileSize++) {
            auto tileInfo = splitHwConvIntoOutChannelsTiles(
                    input->desc().dim(Dim::W), input->desc().dim(Dim::H), curTileSize,
                    curTileSize,
                    stage->attrs().get<int>("kernelSizeX"), stage->attrs().get<int>("kernelSizeY"),
                    stage->attrs().get<int>("kernelStrideX"));

            if (tileInfo.numDescr > 0) {
                auto curNumTiles = divUp(output->desc().dim(Dim::C), curTileSize);
                optTiles.emplace_back(std::make_tuple(curTileSize, tileInfo.cost * curNumTiles));
            }
        }

        //
        // Choose tile with minimal cost
        //

        auto tileSize = output->desc().dim(Dim::C);
        auto numTiles = 1;

        // TODO: switch to SW?
        if (!optTiles.empty()) {
            // Sort by cost.
            std::stable_sort(optTiles.begin(), optTiles.end(),
                [](const OptTile& s1, const OptTile& s2) {
                    return std::get<1>(s1) < std::get<1>(s2);
                });

            double finalCost = 0.0;
            std::tie(tileSize, finalCost) = optTiles[0];

            numTiles = (output->desc().dim(Dim::C) + tileSize - 1) / tileSize;
        }

        //
        // Single tile processing
        //

        if (numTiles == 1) {
            auto constDatas = createWeightableInputsForDepthConv(
                model,
                weights, biases, scales,
                stage,
                tileSize, 0,
                "");

            model->replaceStageInput(stage->inputEdge(1), std::get<0>(constDatas));
            model->replaceStageInput(stage->inputEdge(2), std::get<1>(constDatas));
            model->replaceStageInput(stage->inputEdge(3), std::get<2>(constDatas));

            stage->attrs().set<int>("groupSize", 1);

            continue;
        }

        //
        // Multiple tiles processing
        //

        model->disconnectStage(stage);

        DataVector subInputs(numTiles);
        DataVector subOutputs(numTiles);

        int tileOffset = 0;
        for (int tileInd = 0; tileInd < numTiles; ++tileInd) {
            auto postfix = formatString("@tile=%d/%d", tileInd + 1, numTiles);

            auto curTileSize = tileInd != numTiles - 1 ? tileSize : input->desc().dim(Dim::C) - tileOffset;

            auto inputTileDesc = input->desc();
            inputTileDesc.setDim(Dim::C, curTileSize);

            subInputs[tileInd] = model->duplicateData(
                input,
                postfix,
                inputTileDesc);

            auto outputTileDesc = output->desc();
            outputTileDesc.setDim(Dim::C, curTileSize);

            subOutputs[tileInd] = model->duplicateData(
                output,
                postfix,
                outputTileDesc);

            auto constDatas = createWeightableInputsForDepthConv(
                model,
                weights, biases, scales,
                stage,
                curTileSize, tileOffset,
                postfix);

            auto tileWeights = std::get<0>(constDatas);
            auto tileBiases = std::get<1>(constDatas);
            auto tileScales = std::get<2>(constDatas);

            auto tileStage = model->duplicateStage(
                stage,
                postfix,
                {subInputs[tileInd], tileWeights, tileBiases, tileScales},
                {subOutputs[tileInd]});

            tileStage->attrs().set<int>("groupSize", 1);

            tileOffset += curTileSize;
        }

        _stageBuilder->addSplitStage(
            model,
            stage->name() + "@split",
            stage->origLayer(),
            Dim::C,
            input,
            subInputs);

        _stageBuilder->addConcatStage(
            model,
            stage->name() + "@concat",
            stage->origLayer(),
            Dim::C,
            subOutputs,
            output);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::splitHwDepthConv() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
