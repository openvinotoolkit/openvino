// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/hw/conv_tiling/hw_stage_tiler.hpp>

#include <precision_utils.h>
#include <memory>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <set>

#include <vpu/stages/stub_stage.hpp>
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/utility.hpp>
#include <vpu/utils/attributes_map.hpp>

namespace vpu {

namespace {

constexpr int BIASES_IND = -1;
constexpr int SCALES_IND = -2;

using TileWeightsMap = std::unordered_map<int, Data>;

int getMaxInputChannels(const HwConvTilingPtr& tiling) {
    int max = std::numeric_limits<int>::min();
    for (const auto& tile : tiling->planeTiles) {
        for (const auto& tileOverChannel : tile->channelTiles) {
            max = std::max(max, tileOverChannel->channelStartIndex + tileOverChannel->extendedInputDimC);
        }
    }

    return max;
}

int getMaxOutputChannels(const HwConvTilingPtr& tiling) {
    int max = std::numeric_limits<int>::min();
    for (const auto& tile : tiling->planeTiles) {
        for (const auto& tileOverChannel : tile->channelTiles) {
            max = std::max(max, tileOverChannel->extendedOutputDimC);
        }
    }

    return max;
}

}  // namespace


void HWConvStageTiler::addPoolStage(const HWConvStageIO& stageIO, const HWConvStageOptions& stageOptions) {
    const auto& name = _original->name();
    const auto& orig = _original->origLayer();

    auto hwPoolInput = _model->addNewData(name, stageIO.origOutputDesc);
    hwPoolInput->attrs().copyFrom(stageIO.origOutput->attrs());

    auto hwPoolStage = _model->addNewStage<StubStage>(name + "@Pool", StageType::StubMaxPool, orig,
        {hwPoolInput}, {hwOutput});

    hwPoolStage->attrs().set<int>("kernelSizeX", stageOptions.poolKernelSizeX);
    hwPoolStage->attrs().set<int>("kernelSizeY", stageOptions.poolKernelSizeY);

    hwPoolStage->attrs().set<int>("kernelStrideX", stageOptions.poolKernelStride);
    hwPoolStage->attrs().set<int>("kernelStrideY", stageOptions.poolKernelStride);

    hwPoolStage->attrs().set<int>("padLeft", stageOptions.poolPadLeft);
    hwPoolStage->attrs().set<int>("padRight", stageOptions.poolPadRight);
    hwPoolStage->attrs().set<int>("padTop", stageOptions.poolPadTop);
    hwPoolStage->attrs().set<int>("padBottom", stageOptions.poolPadBottom);

    hwPoolStage->attrs().set<bool>("excludePad", false);

    hwPoolStage->attrs().set<bool>("tryHW", true);

    hwOutput = hwPoolInput;
}

void HWConvStageTiler::expandInput(int numChannels) {
    const auto& name = _original->name();
    const auto& orig = _original->origLayer();

    auto newDesc = hwInput->desc();
    newDesc.setDim(Dim::C, numChannels);

    auto hwInputExtended = _model->duplicateData(hwInput, "@extended", newDesc);

    _stageBuilder->addExpandStage(_model, name + "@expand-input", orig, hwInput, hwInputExtended);
    hwInput = hwInputExtended;
}

Data HWConvStageTiler::createBiases(const HwConvTilingPtr& tiling, const HWConvStageIO& stageIO,
                                    const HWConvStageOptions& stageOptions) {
    const auto origOutputDimC = hwOutput->desc().dim(Dim::C);
    const auto maxExtendedOutputDimC = getMaxOutputChannels(tiling);

    auto& tileWeightsMap = stageIO.origWeights->attrs().getOrSet<TileWeightsMap>("weightsPerTile");
    auto hwBiases = tileWeightsMap[BIASES_IND];
    if (hwBiases == nullptr) {
        if (stageIO.origBiases->usage() == DataUsage::Fake) {
            hwBiases = _model->addFakeData();
        } else {
            auto origBiasesContent = stageIO.origBiases->content();
            IE_ASSERT(origBiasesContent != nullptr);

            auto origBiasesPtr = origBiasesContent->get<fp16_t>();
            IE_ASSERT(origBiasesPtr != nullptr);

            auto hwTileBiasesBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                ie::Precision::FP16,
                {static_cast<size_t>(maxExtendedOutputDimC)},
                ie::Layout::C));
            hwTileBiasesBlob->allocate();

            auto hwTileBiasesBlobPtr = hwTileBiasesBlob->buffer().as<fp16_t*>();
            IE_ASSERT(hwTileBiasesBlobPtr != nullptr);

            std::fill_n(hwTileBiasesBlobPtr, maxExtendedOutputDimC, ie::PrecisionUtils::f32tof16(0.0f));
            std::copy_n(origBiasesPtr, origOutputDimC, hwTileBiasesBlobPtr);

            hwBiases = _model->duplicateData(
                stageIO.origBiases,
                "@HW",
                DataDesc({maxExtendedOutputDimC}),
                ieBlobContent(hwTileBiasesBlob));
        }

        tileWeightsMap[BIASES_IND] = hwBiases;
    }

    return hwBiases;
}

Data HWConvStageTiler::createScales(const HwConvTilingPtr& tiling, const HWConvStageIO& stageIO,
                                    const HWConvStageOptions& stageOptions) {
    const auto origOutputDimC = hwOutput->desc().dim(Dim::C);
    const auto maxExtendedOutputDimC = getMaxOutputChannels(tiling);

    auto& tileWeightsMap = stageIO.origWeights->attrs().getOrSet<TileWeightsMap>("weightsPerTile");
    auto hwScales = tileWeightsMap[SCALES_IND];
    if (hwScales == nullptr) {
        if (stageIO.origScales->usage() == DataUsage::Fake) {
            if (tiling->socTiles == 1 && stageOptions.reluScale != 1.0f) {
                hwScales = _model->addConstData(
                    _original->name() + "@scales",
                    DataDesc({maxExtendedOutputDimC}),
                    replicateContent(stageOptions.reluScale, maxExtendedOutputDimC));
            } else {
                hwScales = _model->addFakeData();
            }
        } else {
            auto origScalesContent = stageIO.origScales->content();
            IE_ASSERT(origScalesContent != nullptr);

            auto origScalesPtr = origScalesContent->get<fp16_t>();
            IE_ASSERT(origScalesPtr != nullptr);

            auto hwTileScalesBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                ie::Precision::FP16,
                {static_cast<size_t>(maxExtendedOutputDimC)},
                ie::Layout::C));
            hwTileScalesBlob->allocate();

            auto hwTileScalesBlobPtr = hwTileScalesBlob->buffer().as<fp16_t*>();
            IE_ASSERT(hwTileScalesBlobPtr != nullptr);

            std::fill_n(hwTileScalesBlobPtr, maxExtendedOutputDimC, ie::PrecisionUtils::f32tof16(0.0f));
            IE_ASSERT(maxExtendedOutputDimC >= origOutputDimC);
            std::copy_n(origScalesPtr, origOutputDimC, hwTileScalesBlobPtr);

            hwScales = _model->duplicateData(
                stageIO.origScales,
                "@HW",
                DataDesc({maxExtendedOutputDimC}),
                ieBlobContent(hwTileScalesBlob));

            if (tiling->socTiles == 1 && stageOptions.reluScale != 1.0f) {
                hwScales = _model->duplicateData(
                    hwScales,
                    "@HW",
                    DataDesc(),
                    scaleContent(hwScales->content(), stageOptions.reluScale));
            }
        }

        tileWeightsMap[SCALES_IND] = hwScales;
    }

    return hwScales;
}

Data HWConvStageTiler::createOutputTile(const HwConvPlaneTilePtr& planeTile, const std::string& tilePostfix,
                                        const HwConvTilingPtr& tiling) {
    Data hwOutputPlaneTile;
    if (tiling->sohTiles == 1 && tiling->sowTiles == 1) {
        hwOutputPlaneTile = hwOutput;
    } else {
        auto newDesc = hwOutput->desc();
        newDesc.setDim(Dim::W, planeTile->widthInfo.outputEndIndex - planeTile->widthInfo.outputStartIndex);
        newDesc.setDim(Dim::H, planeTile->heightInfo.outputEndIndex - planeTile->heightInfo.outputStartIndex);

        hwOutputPlaneTile = _model->duplicateData(hwOutput, tilePostfix, newDesc);

        hwOutputTiles.emplace_back(hwOutputPlaneTile);
        hwOutputTilesOffsets.emplace_back(DimValues({
            {Dim::W, planeTile->widthInfo.outputStartIndex},
            {Dim::H, planeTile->heightInfo.outputStartIndex}}));
    }

    if ((planeTile->widthInfo.outputStartIndex * sizeof(fp16_t)) % 16 != 0) {
        auto hwOutputPlaneTileAligned = _model->duplicateData(hwOutputPlaneTile, "@aligned");

        _stageBuilder->addCopyStage(
            _model,
            _original->name() + tilePostfix + "@align-output-ptr",
            _original->origLayer(),
            hwOutputPlaneTileAligned,
            hwOutputPlaneTile,
            "HWConvTiler::output");

        hwOutputPlaneTile = hwOutputPlaneTileAligned;
    }

    return hwOutputPlaneTile;
}

Data HWConvStageTiler::createInputTile(const HwConvPlaneTilePtr& planeTile, const HwConvChannelTilePtr& channelTile,
                                       const std::string& tilePostfix, const HwConvTilingPtr& tiling) {
    Data hwInputTile;
    if (tiling->sohTiles == 1 && tiling->sowTiles == 1 && tiling->socTiles == 1) {
        hwInputTile = hwInput;
    } else {
        auto newDesc = hwInput->desc();
        newDesc.setDim(Dim::W, planeTile->widthInfo.inputWithJunk);
        newDesc.setDim(Dim::H, planeTile->heightInfo.inputWithJunk);
        newDesc.setDim(Dim::C, channelTile->extendedInputDimC);

        hwInputTile = _model->duplicateData(hwInput, tilePostfix, newDesc);

        hwInputTiles.emplace_back(hwInputTile);
        hwInputTilesOffsets.emplace_back(DimValues({
            {Dim::W, planeTile->widthInfo.inputStartIndex},
            {Dim::H, planeTile->heightInfo.inputStartIndex},
            {Dim::C, channelTile->channelStartIndex}}));
    }

    if ((planeTile->widthInfo.inputStartIndex * sizeof(fp16_t)) % 16 != 0) {
        auto hwInputTileAligned = _model->duplicateData(hwInputTile, "@aligned");

        _stageBuilder->addCopyStage(
            _model,
            _original->name() + tilePostfix + "@align-input-ptr",
            _original->origLayer(),
            hwInputTile,
            hwInputTileAligned,
            "HWConvTiler::input");

        hwInputTile = hwInputTileAligned;
    }

    return hwInputTile;
}

Data HWConvStageTiler::reducePartialOverChannelsOutputs(const Data& hwOutputPlaneTile,
                                                        const HwConvPlaneTilePtr& planeTile,
                                                        const HwConvChannelTilePtr& channelTile,
                                                        const std::string& channelTilePostfix,
                                                        const std::string& tilePostfix,
                                                        const HWConvStageOptions& stageOptions,
                                                        const HwConvTilingPtr& tiling,
                                                        Data& prevPartialSum) {
    auto hwOutputTile = hwOutputPlaneTile;
    if (tiling->socTiles > 1) {
        auto hwConvPartialOutput = _model->duplicateData(hwOutputTile, channelTilePostfix + "@partial");

        if (channelTile->socInd == 0) {
            prevPartialSum = hwConvPartialOutput;
        } else {
            auto sumPartialOutput = hwOutputTile;
            if (channelTile->socInd < tiling->socTiles - 1 || stageOptions.withReLU || stageOptions.withClamp) {
                sumPartialOutput = _model->duplicateData(hwOutputTile, channelTilePostfix + "@accum");
            }

            _stageBuilder->addSumStage(
                _model,
                _original->name() + tilePostfix + "@accum",
                _original->origLayer(),
                prevPartialSum, hwConvPartialOutput,
                sumPartialOutput);

            if (channelTile->socInd == tiling->socTiles - 1 && stageOptions.withReLU) {
                _stageBuilder->addReLUStage(
                    _model,
                    _original->name() + tilePostfix + "@ReLU",
                    _original->origLayer(),
                    stageOptions.negativeSlope,
                    sumPartialOutput,
                    hwOutputTile);
            }

            if (channelTile->socInd == tiling->socTiles - 1 && stageOptions.withClamp) {
                _stageBuilder->addClampStage(
                    _model,
                    _original->name() + tilePostfix + "@Clamp",
                    _original->origLayer(),
                    0.0,
                    stageOptions.clampMax,
                    sumPartialOutput,
                    hwOutputTile);
            }

            prevPartialSum = sumPartialOutput;
        }

        hwOutputTile = hwConvPartialOutput;
    }

    if (planeTile->heightInfo.outputJunkBefore != 0 || planeTile->heightInfo.outputJunkAfter != 0 ||
        planeTile->widthInfo.outputJunkBefore != 0 || planeTile->widthInfo.outputJunkAfter != 0) {
        auto newDesc = hwOutputTile->desc();
        newDesc.setDim(Dim::W, planeTile->widthInfo.outputWithJunk);
        newDesc.setDim(Dim::H, planeTile->heightInfo.outputWithJunk);

        auto hwOutputTileWithJunk = _model->duplicateData(hwOutputTile, "@with-junk", newDesc);

        DimValues innerOffset;
        innerOffset.set(Dim::W, planeTile->widthInfo.outputJunkBefore);
        innerOffset.set(Dim::H, planeTile->heightInfo.outputJunkBefore);

        _stageBuilder->addShrinkStage(
            _model,
            _original->name() + tilePostfix + "@remove-junk",
            _original->origLayer(),
            hwOutputTileWithJunk,
            hwOutputTile,
            innerOffset);

        hwOutputTile = hwOutputTileWithJunk;
    }

    return hwOutputTile;
}

Data HWConvStageTiler::createTileWeights(const HwConvChannelTilePtr& channelTile, const std::string& channelTilePostfix,
                                        const HWConvStageIO& io, const HWConvStageOptions& options) {
    auto& tileWeightsMap = io.origWeights->attrs().getOrSet<TileWeightsMap>("weightsPerTile");
    auto hwTileWeights = tileWeightsMap[channelTile->socInd];

    if (hwTileWeights == nullptr) {
        const int vector_size = 8;
        const auto descriptor = DataDesc{{
            vector_size,
            options.kernelSizeX * options.kernelSizeY,
            channelTile->extendedInputDimC,
            channelTile->extendedOutputDimC / vector_size
         }};

        const auto content = std::make_shared<HwWeightsContent>(
            io.origWeights->content(),
            io.origWeights->desc(),
            channelTile->numInputChannels,
            channelTile->channelStartIndex);

        hwTileWeights = _model->duplicateData(io.origWeights, "@HW" + channelTilePostfix, descriptor, content);
        tileWeightsMap[channelTile->socInd] = hwTileWeights;
    }

    return hwTileWeights;
}

Data HWConvStageTiler::createTileBiases(const Data& hwBiases, const HwConvChannelTilePtr& channelTile) {
    return channelTile->socInd > 0 ? _model->addFakeData() : hwBiases;
}

void HWConvStageTiler::createHWStageForTile(const Data& hwInputTile,
                                            const Data& hwTileWeights,
                                            const Data& hwTileBiases,
                                            const Data& hwScales,
                                            const Data& hwOutputTile,
                                            bool tileStageWithPool,
                                            const HwConvChannelTilePtr& channelTile,
                                            const HWConvStageOptions& stageOptions,
                                            const std::string& tilePostfix,
                                            const HwConvTilingPtr& tiling) {
    auto hwOutputTileDims = hwOutputTile->desc().dims();
    if (tileStageWithPool) {
        hwOutputTileDims.set(
            Dim::W,
            hwOutputTileDims[Dim::W] * stageOptions.poolKernelStride -
            stageOptions.poolPadLeft - stageOptions.poolPadRight);
        hwOutputTileDims.set(
            Dim::H,
            hwOutputTileDims[Dim::H] * stageOptions.poolKernelStride -
            stageOptions.poolPadTop - stageOptions.poolPadBottom);
    }

    auto hwStage = _model->addNewStage<MyriadXHwStage>(
        _original->name() + tilePostfix,
        StageType::MyriadXHwOp,
        _original->origLayer(),
        {hwInputTile, hwTileWeights, hwTileBiases, hwScales},
        {hwOutputTile});

    hwStage->attrs().set<HwOpType>("hwOpType", tileStageWithPool ? HwOpType::CONV_POOL : HwOpType::CONV);

    hwStage->attrs().set<int>("kernelSizeX", stageOptions.kernelSizeX);
    hwStage->attrs().set<int>("kernelSizeY", stageOptions.kernelSizeY);
    hwStage->attrs().set<int>("kernelStride", stageOptions.kernelStride);

    if (tileStageWithPool) {
        hwStage->attrs().set<int>("poolKernelSizeX", stageOptions.poolKernelSizeX);
        hwStage->attrs().set<int>("poolKernelSizeY", stageOptions.poolKernelSizeY);
    }

    const auto hwPad = getHwPaddingInfo(
        hwInputTile->desc().dims(), hwOutputTileDims,
        stageOptions.kernelSizeX, stageOptions.kernelSizeY,
        stageOptions.kernelStride, stageOptions.kernelStride,
        stageOptions.padLeft, stageOptions.padTop);

    hwStage->attrs().set<HwPaddingInfo>("pad", hwPad);

    hwStage->attrs().set<HwConvTileInfo>("tiling", channelTile->finalTiles);

    if (tiling->socTiles > 1) {
        hwStage->attrs().set<bool>("withReLU", false);
        hwStage->attrs().set<bool>("withClamp", false);
    } else {
        hwStage->attrs().set<bool>("withReLU", stageOptions.withReLU);
        hwStage->attrs().set<uint32_t>("a0", stageOptions.a0);
        hwStage->attrs().set<uint32_t>("a1", stageOptions.a1);
        hwStage->attrs().set<float>("negativeSlope", stageOptions.negativeSlope);

        hwStage->attrs().set<bool>("withClamp", stageOptions.withClamp);
        hwStage->attrs().set<float>("clampMax", stageOptions.clampMax);
    }

    hwStage->attrs().set<float>("scaleFactor", stageOptions.scaleFactor);
}

HWConvStageTiler::HWConvStageTiler(const HWConvStageOptions& stageOptions, const HWConvStageIO& stageIO,
                                   const Model& model, const Stage& origStage,
                                   const StageBuilder::Ptr& stageBuilder, const HwConvTilingPtr& tiling,
                                   bool makeExplicitPoolStage) {
    hwInput = stageIO.origInput;
    hwOutput = stageIO.origOutput;
    _stageBuilder = stageBuilder;
    _model = model;
    _original = origStage;

    bool tileStageWithPool = stageOptions.withPool;
    if (makeExplicitPoolStage) {
        addPoolStage(stageIO, stageOptions);
        tileStageWithPool = false;
    }

    const auto totalExtendedInputDimC = getMaxInputChannels(tiling);

    if (totalExtendedInputDimC > hwInput->desc().dim(Dim::C)) {
        expandInput(totalExtendedInputDimC);
    }

    const auto hwBiases = createBiases(tiling, stageIO, stageOptions);
    const auto hwScales = createScales(tiling, stageIO, stageOptions);

    //
    // Create HW tiles
    //

    hwInputTiles.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwInputTilesOffsets.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwOutputTiles.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwOutputTilesOffsets.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);

    for (const auto& planeTile : tiling->planeTiles) {
        const auto planeTilePostfix = getPlaneTilePostfix(planeTile);

        const auto hwOutputPlaneTile = createOutputTile(planeTile, planeTilePostfix, tiling);

        Data prevPartialSum;
        for (const auto& channelTile : planeTile->channelTiles) {
            const auto channelTilePostfix = getChannelTilePostfix(channelTile);
            const auto tilePostfix = planeTilePostfix + channelTilePostfix;
            const auto hwInputTile = createInputTile(planeTile, channelTile, tilePostfix, tiling);

            const auto hwOutputTile = reducePartialOverChannelsOutputs(
                hwOutputPlaneTile,
                planeTile,
                channelTile,
                channelTilePostfix,
                tilePostfix,
                stageOptions,
                tiling,
                prevPartialSum);

            const auto hwTileWeights = createTileWeights(channelTile, channelTilePostfix, stageIO, stageOptions);

            const auto hwTileBiases = createTileBiases(hwBiases, channelTile);

            createHWStageForTile(hwInputTile, hwTileWeights, hwTileBiases, hwScales, hwOutputTile, tileStageWithPool,
                                 channelTile, stageOptions, tilePostfix, tiling);
        }
    }
}

}  // namespace vpu
