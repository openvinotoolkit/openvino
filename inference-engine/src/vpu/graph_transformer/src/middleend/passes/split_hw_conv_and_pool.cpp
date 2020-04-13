// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vector>
#include <set>
#include <memory>
#include <array>

#include <vpu/compile_env.hpp>
#include <vpu/middleend/hw/utility.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(splitHwConvAndPool);

    const auto& env = CompileEnv::get();
    const auto cmxLimit = tilingCMXLimit(env.resources.numCMXSlices);

    for (const auto& convStage : model->getStages()) {
        if (convStage == nullptr) {
            continue;
        }

        if (convStage->type() != StageType::StubConv) {
            continue;
        }

        auto convHW = convStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!convHW) {
            continue;
        }

        auto convInput = convStage->input(0);
        auto convWeights = convStage->input(1);
        auto convBiases = convStage->input(2);
        auto convScales = convStage->input(3);
        auto convOutput = convStage->output(0);

        if (convOutput->usage() != DataUsage::Intermediate) {
            continue;
        }

        // TODO : better estimation?
        auto outBufSize = calculateHwBufferSize(convOutput->desc().dims());
        if (outBufSize <= cmxLimit) {
            continue;
        }

        if (convOutput->numConsumers() != 1) {
            continue;
        }

        auto poolStage = convOutput->singleConsumer();
        if (poolStage->type() != StageType::StubAvgPool &&
            poolStage->type() != StageType::StubMaxPool) {
            continue;
        }

        auto poolHW = poolStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!poolHW) {
            continue;
        }

        auto convKernelSizeX = convStage->attrs().get<int>("kernelSizeX");
        auto convKernelSizeY = convStage->attrs().get<int>("kernelSizeY");

        auto poolOutput = poolStage->output(0);

        // TODO : better estimation?
        int tileSize = 0u;
        std::array<int, 3> TILE_SIZE_CANDIDATES{{128, 64, 32}};
        for (auto curTileSize : TILE_SIZE_CANDIDATES) {
            if (convOutput->desc().dim(Dim::C) >= curTileSize &&
                convOutput->desc().dim(Dim::C) % curTileSize == 0) {
                DimValues curOutDims;
                curOutDims.set(Dim::W, convOutput->desc().dim(Dim::W));
                curOutDims.set(Dim::H, convOutput->desc().dim(Dim::H));
                curOutDims.set(Dim::C, curTileSize);

                auto curOutBufSize = calculateHwBufferSize(curOutDims);
                if (curOutBufSize <= cmxLimit) {
                    tileSize = curTileSize;
                    break;
                }
            }
        }

        if (tileSize == 0)
            continue;

        auto numTiles = (convOutput->desc().dim(Dim::C) + tileSize - 1) / tileSize;

        model->disconnectStage(convStage);
        model->disconnectStage(poolStage);

        DataVector subOutputs(numTiles);

        int tileOffset = 0;
        for (int tileInd = 0; tileInd < numTiles; ++tileInd) {
            auto postfix = formatString("@tile=%d/%d", tileInd + 1, numTiles);

            auto curTileSize = tileInd != numTiles - 1 ? tileSize : convOutput->desc().dim(Dim::C) - tileOffset;

            auto convOutputTileDesc = convOutput->desc();
            convOutputTileDesc.setDim(Dim::C, curTileSize);

            auto convOutputTile = model->duplicateData(
                convOutput,
                postfix,
                convOutputTileDesc);

            auto poolOutputTileDesc = poolOutput->desc();
            poolOutputTileDesc.setDim(Dim::C, curTileSize);

            auto poolOutputTile = model->duplicateData(
                poolOutput,
                postfix,
                poolOutputTileDesc);

            Data tileWeights;
            {
                auto content = convWeights->content();
                IE_ASSERT(content != nullptr);

                auto origWeights = content->get<fp16_t>();
                IE_ASSERT(origWeights != nullptr);

                auto kernWxH = convKernelSizeX * convKernelSizeY;
                size_t newWeightsSize = kernWxH * convInput->desc().dim(Dim::C) * tileSize;

                auto newWeightsBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                    ie::Precision::FP16,
                    {newWeightsSize},
                    ie::Layout::C));
                newWeightsBlob->allocate();

                auto inPtr = origWeights + kernWxH * convInput->desc().dim(Dim::C) * tileInd * tileSize;
                auto outPtr = newWeightsBlob->buffer().as<fp16_t*>();

                std::copy_n(inPtr, newWeightsSize, outPtr);

                tileWeights = model->duplicateData(
                    convWeights,
                    postfix,
                    DataDesc({convKernelSizeX, convKernelSizeY, convInput->desc().dim(Dim::C), tileSize}),
                    ieBlobContent(newWeightsBlob));
            }

            auto tileBiases = convBiases;
            if (convBiases->usage() != DataUsage::Fake) {
                auto content = convBiases->content();
                IE_ASSERT(content != nullptr);

                auto origBiases = content->get<fp16_t>();
                IE_ASSERT(origBiases != nullptr);

                auto newBiasesBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                    ie::Precision::FP16,
                    {static_cast<size_t>(tileSize)},
                    ie::Layout::C));
                newBiasesBlob->allocate();

                auto inPtr = origBiases + tileInd * tileSize;
                auto outPtr = newBiasesBlob->buffer().as<fp16_t*>();

                std::copy_n(inPtr, tileSize, outPtr);

                tileBiases = model->duplicateData(
                    convBiases,
                    postfix,
                    DataDesc({tileSize}),
                    ieBlobContent(newBiasesBlob));
            }

            auto tileScales = convScales;
            if (convScales->usage() != DataUsage::Fake) {
                auto content = convScales->content();
                IE_ASSERT(content != nullptr);

                auto origScales = content->get<fp16_t>();
                IE_ASSERT(origScales != nullptr);

                auto newScalesBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                    ie::Precision::FP16,
                    {static_cast<size_t>(tileSize)},
                    ie::Layout::C));
                newScalesBlob->allocate();

                auto inPtr = origScales + tileInd * tileSize;
                auto outPtr = newScalesBlob->buffer().as<fp16_t*>();

                std::copy_n(inPtr, tileSize, outPtr);

                 tileScales = model->duplicateData(
                    convScales,
                    postfix,
                    DataDesc({tileSize}),
                    ieBlobContent(newScalesBlob));
            }

            model->duplicateStage(
                convStage,
                postfix,
                {convInput, tileWeights, tileBiases, tileScales},
                {convOutputTile});

            model->duplicateStage(
                poolStage,
                postfix,
                {convOutputTile},
                {poolOutputTile});

            subOutputs[tileInd] = poolOutputTile;

            tileOffset += curTileSize;
        }

        _stageBuilder->addConcatStage(
            model,
            poolStage->name() + "@concat",
            poolStage->origLayer(),
            Dim::C,
            subOutputs,
            poolOutput);

        model->removeStage(convStage);
        model->removeStage(poolStage);
    }
}

}  // namespace

Pass::Ptr PassManager::splitHwConvAndPool() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
