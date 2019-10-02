// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <vpu/passes/hw_conv_tiling/hw_stage_tiler.hpp>

#include <precision_utils.h>
#include <tuple>
#include <utility>
#include <memory>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/stub_stage.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/tiling.hpp>
#include <vpu/hw/utility.hpp>
#include <vpu/utils/attributes_map.hpp>

namespace vpu {

const int BIASES_IND = -1;
const int SCALES_IND = -2;

using TileWeightsMap = std::unordered_map<int, Data>;

HWConvStageTiler::HWConvStageTiler(const HWConvStageOptions &so, const HWConvStageIO &sio,
                           const Model::Ptr &model, const Handle <StageNode> &origStage,
                           const StageBuilder::Ptr &_stageBuilder, const HwConvTilingPtr &tiling,
                       const bool makeExplicitPoolStage) {
    hwInput = sio.origInput;
    hwOutput = sio.origOutput;

    //
    // Create explicit pool stage if tiling with pool is not possible
    //
    bool tileStageWithPool = so.withPool;
    if (makeExplicitPoolStage) {
        auto hwPoolInput = model->addNewData(
                origStage->name(),
                sio.origOutputDesc);
        hwPoolInput->attrs().copyFrom(sio.origOutput->attrs());

        auto hwPoolStage = model->addNewStage<StubStage>(
                origStage->name() + "@Pool",
                StageType::StubMaxPool,
                origStage->origLayer(),
                {hwPoolInput},
                {hwOutput});

        hwPoolStage->attrs().set<int>("kernelSizeX", so.poolKernelSizeX);
        hwPoolStage->attrs().set<int>("kernelSizeY", so.poolKernelSizeY);

        hwPoolStage->attrs().set<int>("kernelStrideX", so.poolKernelStride);
        hwPoolStage->attrs().set<int>("kernelStrideY", so.poolKernelStride);

        hwPoolStage->attrs().set<int>("padLeft", so.poolPadLeft);
        hwPoolStage->attrs().set<int>("padRight", so.poolPadRight);
        hwPoolStage->attrs().set<int>("padTop", so.poolPadTop);
        hwPoolStage->attrs().set<int>("padBottom", so.poolPadBottom);

        hwPoolStage->attrs().set<bool>("excludePad", false);

        hwPoolStage->attrs().set<bool>("tryHW", true);

        hwOutput = hwPoolInput;
        tileStageWithPool = false;
    }

    //
    // Expand input/output if needed
    //

    int totalExtendedInputDimC = 0;
    int maxExtendedOutputDimC = 0;

    for (const auto& planeTile : tiling->planeTiles) {
        for (const auto& channelTile : planeTile->channelTiles) {
            totalExtendedInputDimC = std::max(totalExtendedInputDimC, channelTile->channelStartIndex + channelTile->extendedInputDimC);
            maxExtendedOutputDimC = std::max(maxExtendedOutputDimC, channelTile->extendedOutputDimC);
        }
    }

    auto origOutputDimC = hwOutput->desc().dim(Dim::C);

    if (totalExtendedInputDimC > hwInput->desc().dim(Dim::C)) {
        auto newDesc = hwInput->desc();
        newDesc.setDim(Dim::C, totalExtendedInputDimC);

        auto hwInputExtended = model->duplicateData(
                hwInput,
                "@extended",
                newDesc);

        _stageBuilder->addExpandStage(
                model,
                origStage->name() + "@expand-input",
                origStage->origLayer(),
                hwInput,
                hwInputExtended);

        hwInput = hwInputExtended;
    }

    //
    // Create HW biases
    //

    auto& tileWeightsMap = sio.origWeights->attrs().getOrSet<TileWeightsMap>("weightsPerTile", TileWeightsMap());
    auto hwBiases = tileWeightsMap[BIASES_IND];
    if (hwBiases == nullptr) {
        if (sio.origBiases->usage() == DataUsage::Fake) {
            hwBiases = model->addFakeData();
        } else {
            auto origBiasesContent = sio.origBiases->content();
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

            hwBiases = model->duplicateData(
                    sio.origBiases,
                    "@HW",
                    DataDesc({maxExtendedOutputDimC}),
                    ieBlobContent(hwTileBiasesBlob));

            if (so.scaleFactor != 1.0f) {
                auto hwBiasesScaled = model->duplicateData(
                        hwBiases,
                        formatString("@SCALE=%f", so.scaleFactor),
                        hwBiases->desc(),
                        scaleContent(hwBiases->content(), so.scaleFactor));
                hwBiasesScaled->attrs().getOrSet<float>("scaleFactor", 1.0f) *= so.scaleFactor;

                hwBiases = hwBiasesScaled;
            }
        }

        tileWeightsMap[BIASES_IND] = hwBiases;
    }

    //
    // Create HW scales
    //

    auto hwScales = tileWeightsMap[SCALES_IND];
    if (hwScales == nullptr) {
        float fullScale = 1.0f / so.scaleFactor;
        if (tiling->socTiles == 1 && so.reluScale != 1.0f) {
            fullScale *= so.reluScale;
        }

        if (fullScale == 1.0f) {
            hwScales = model->addFakeData();
        } else {
            hwScales = model->addConstData(
                    origStage->name() + "@scales",
                    DataDesc({maxExtendedOutputDimC}),
                    replicateContent(fullScale, maxExtendedOutputDimC));
        }

        tileWeightsMap[SCALES_IND] = hwScales;
    }

    //
    // Create HW tiles
    //

    hwInputTiles.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwInputTilesOffsets.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwOutputTiles.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwOutputTilesOffsets.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);

    for (const auto& planeTile : tiling->planeTiles) {
        auto planeTilePostfix = getPlaneTilePostfix(planeTile);

        //
        // Create output tile
        //

        Data hwOutputPlaneTile;

        if (tiling->sohTiles == 1 && tiling->sowTiles == 1) {
            hwOutputPlaneTile = hwOutput;
        } else {
            auto newDesc = hwOutput->desc();
            newDesc.setDim(Dim::W, planeTile->widthInfo.outputEndIndex - planeTile->widthInfo.outputStartIndex);
            newDesc.setDim(Dim::H, planeTile->heightInfo.outputEndIndex - planeTile->heightInfo.outputStartIndex);

            hwOutputPlaneTile = model->duplicateData(
                    hwOutput,
                    planeTilePostfix,
                    newDesc);

            hwOutputTiles.emplace_back(hwOutputPlaneTile);
            hwOutputTilesOffsets.emplace_back(
                    DimValues({
                                      {Dim::W, planeTile->widthInfo.outputStartIndex},
                                      {Dim::H, planeTile->heightInfo.outputStartIndex}
                              }));
        }

        //
        // Add alignment to output tile if needed
        //

        if ((planeTile->widthInfo.outputStartIndex * sizeof(fp16_t)) % 16 != 0) {
            auto hwOutputPlaneTileAligned = model->duplicateData(
                    hwOutputPlaneTile,
                    "@aligned");

            _stageBuilder->addCopyStage(
                    model,
                    origStage->name() + planeTilePostfix + "@align-output-ptr",
                    origStage->origLayer(),
                    hwOutputPlaneTileAligned,
                    hwOutputPlaneTile,
                    "HWConvTiler::output");

            hwOutputPlaneTile = hwOutputPlaneTileAligned;
        }

        Data prevPartialSum;

        for (const auto& channelTile : planeTile->channelTiles) {
            auto channelTilePostfix = getChannelTilePostfix(channelTile);

            auto tilePostfix = planeTilePostfix + channelTilePostfix;

            auto hwOutputTile = hwOutputPlaneTile;

            //
            // Create input tile
            //

            Data hwInputTile;

            if (tiling->sohTiles == 1 && tiling->sowTiles == 1 && tiling->socTiles == 1) {
                hwInputTile = hwInput;
            } else {
                auto newDesc = hwInput->desc();
                newDesc.setDim(Dim::W, planeTile->widthInfo.inputWithJunk);
                newDesc.setDim(Dim::H, planeTile->heightInfo.inputWithJunk);
                newDesc.setDim(Dim::C, channelTile->extendedInputDimC);

                hwInputTile = model->duplicateData(
                        hwInput,
                        tilePostfix,
                        newDesc);

                hwInputTiles.emplace_back(hwInputTile);
                hwInputTilesOffsets.emplace_back(
                        DimValues({
                                          {Dim::W, planeTile->widthInfo.inputStartIndex},
                                          {Dim::H, planeTile->heightInfo.inputStartIndex},
                                          {Dim::C, channelTile->channelStartIndex}
                                  }));
            }

            //
            // Add alignment to input tile if needed
            //

            if ((planeTile->widthInfo.inputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                auto hwInputTileAligned = model->duplicateData(
                        hwInputTile,
                        "@aligned");

                _stageBuilder->addCopyStage(
                        model,
                        origStage->name() + tilePostfix + "@align-input-ptr",
                        origStage->origLayer(),
                        hwInputTile,
                        hwInputTileAligned,
                        "HWConvTiler::input");

                hwInputTile = hwInputTileAligned;
            }

            //
            // Process partial output for split-over-channels
            //

            if (tiling->socTiles > 1) {
                auto hwConvPartialOutput = model->duplicateData(
                        hwOutputTile,
                        channelTilePostfix + "@partial");

                if (channelTile->socInd == 0) {
                    prevPartialSum = hwConvPartialOutput;
                } else {
                    auto sumPartialOutput = hwOutputTile;
                    if (channelTile->socInd < tiling->socTiles - 1 || so.withReLU || so.withClamp) {
                        sumPartialOutput = model->duplicateData(
                                hwOutputTile,
                                channelTilePostfix + "@accum");
                    }

                    _stageBuilder->addSumStage(
                            model,
                            origStage->name() + tilePostfix + "@accum",
                            origStage->origLayer(),
                            prevPartialSum, hwConvPartialOutput,
                            sumPartialOutput);

                    if (channelTile->socInd == tiling->socTiles - 1 && so.withReLU) {
                        _stageBuilder->addReLUStage(
                                model,
                                origStage->name() + tilePostfix + "@ReLU",
                                origStage->origLayer(),
                                so.negativeSlope,
                                sumPartialOutput,
                                hwOutputTile);
                    }

                    if (channelTile->socInd == tiling->socTiles - 1 && so.withClamp) {
                        _stageBuilder->addClampStage(
                                model,
                                origStage->name() + tilePostfix + "@Clamp",
                                origStage->origLayer(),
                                0.0,
                                so.clampMax,
                                sumPartialOutput,
                                hwOutputTile);
                    }

                    prevPartialSum = sumPartialOutput;
                }

                hwOutputTile = hwConvPartialOutput;
            }

            //
            // Process output junk if needed
            //

            if (planeTile->heightInfo.outputJunkBefore != 0 ||
                planeTile->heightInfo.outputJunkAfter != 0 ||
                planeTile->widthInfo.outputJunkBefore != 0 ||
                planeTile->widthInfo.outputJunkAfter != 0) {
                auto newDesc = hwOutputTile->desc();
                newDesc.setDim(Dim::W, planeTile->widthInfo.outputWithJunk);
                newDesc.setDim(Dim::H, planeTile->heightInfo.outputWithJunk);

                auto hwOutputTileWithJunk = model->duplicateData(
                        hwOutputTile,
                        "@with-junk",
                        newDesc);

                DimValues innerOffset;
                innerOffset.set(Dim::W, planeTile->widthInfo.outputJunkBefore);
                innerOffset.set(Dim::H, planeTile->heightInfo.outputJunkBefore);

                _stageBuilder->addShrinkStage(
                        model,
                        origStage->name() + tilePostfix + "@remove-junk",
                        origStage->origLayer(),
                        hwOutputTileWithJunk,
                        hwOutputTile,
                        innerOffset);

                hwOutputTile = hwOutputTileWithJunk;
            }

            //
            // Create tile weights
            //

            auto hwTileWeights = tileWeightsMap[channelTile->socInd];

            if (hwTileWeights == nullptr) {
                hwTileWeights = model->duplicateData(
                        sio.origWeights,
                        "@HW" + channelTilePostfix,
                        DataDesc({8, so.kernelSizeX * so.kernelSizeY, channelTile->extendedInputDimC, channelTile->extendedOutputDimC / 8}),
                        std::make_shared<HwWeightsContent>(
                                sio.origWeights->content(),
                                sio.origWeights->desc(),
                                channelTile->numInputChannels,
                                channelTile->channelStartIndex));

                if (so.scaleFactor != 1.0f) {
                    auto hwTileWeightsScaled = model->duplicateData(
                            hwTileWeights,
                            formatString("@SCALE=%f", so.scaleFactor),
                            hwTileWeights->desc(),
                            scaleContent(hwTileWeights->content(), so.scaleFactor));
                    hwTileWeightsScaled->attrs().getOrSet<float>("scaleFactor", 1.0f) *= so.scaleFactor;

                    hwTileWeights = hwTileWeightsScaled;
                }

                tileWeightsMap[channelTile->socInd] = hwTileWeights;
            }

            //
            // Create tile biases
            //

            Data hwTileBiases;

            if (channelTile->socInd > 0) {
                hwTileBiases = model->addFakeData();
            } else {
                hwTileBiases = hwBiases;
            }

            //
            // Create HW stage for tile
            //

            auto hwOutputTileDims = hwOutputTile->desc().dims();
            if (tileStageWithPool) {
                hwOutputTileDims.set(Dim::W, hwOutputTileDims[Dim::W] * so.poolKernelStride - so.poolPadLeft - so.poolPadRight);
                hwOutputTileDims.set(Dim::H, hwOutputTileDims[Dim::H] * so.poolKernelStride - so.poolPadTop - so.poolPadBottom);
            }

            auto hwPad = getHwPaddingInfo(
                    hwInputTile->desc().dims(), hwOutputTileDims,
                    so.kernelSizeX, so.kernelSizeY,
                    so.kernelStride, so.kernelStride,
                    so.padLeft, so.padTop);

            auto hwStage = model->addNewStage<MyriadXHwStage>(
                    origStage->name() + tilePostfix,
                    StageType::MyriadXHwOp,
                    origStage->origLayer(),
                    {hwInputTile, hwTileWeights, hwTileBiases, hwScales},
                    {hwOutputTile});

            hwStage->attrs().set<HwOpType>("hwOpType", tileStageWithPool ? HwOpType::CONV_POOL : HwOpType::CONV);

            hwStage->attrs().set<int>("kernelSizeX", so.kernelSizeX);
            hwStage->attrs().set<int>("kernelSizeY", so.kernelSizeY);
            hwStage->attrs().set<int>("kernelStride", so.kernelStride);

            if (tileStageWithPool) {
                hwStage->attrs().set<int>("poolKernelSizeX", so.poolKernelSizeX);
                hwStage->attrs().set<int>("poolKernelSizeY", so.poolKernelSizeY);
            }

            hwStage->attrs().set<HwPaddingInfo>("pad", hwPad);

            hwStage->attrs().set<HwConvTileInfo>("tiling", channelTile->finalTiles);

            if (tiling->socTiles > 1) {
                hwStage->attrs().set<bool>("withReLU", false);
                hwStage->attrs().set<bool>("withClamp", false);
            } else {
                hwStage->attrs().set<bool>("withReLU", so.withReLU);
                hwStage->attrs().set<uint32_t>("a0", so.a0);
                hwStage->attrs().set<uint32_t>("a1", so.a1);
                hwStage->attrs().set<float>("negativeSlope", so.negativeSlope);

                hwStage->attrs().set<bool>("withClamp", so.withClamp);
                hwStage->attrs().set<float>("clampMax", so.clampMax);
            }

            hwStage->attrs().set<float>("scaleFactor", so.scaleFactor);
        }
    }
}

}  // namespace vpu
