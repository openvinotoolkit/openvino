// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <cmath>

#include <tuple>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <utility>
#include <vector>
#include <memory>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/stub_stage.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/tiling.hpp>
#include <vpu/hw/utility.hpp>

namespace vpu {

namespace {

const int CHANS_PER_DESCR = 16;

HwPoolTileInfo splitPooling(int outZ) {
    HwPoolTileInfo tiles;
    tiles.mode = HwOpMode::MODE_16_16;
    tiles.numDescr = (outZ + CHANS_PER_DESCR - 1) / CHANS_PER_DESCR;
    tiles.chansPerDescr = CHANS_PER_DESCR;
    return tiles;
}

class Optimizer final {
public:
    Optimizer(const std::string& stageName,
              const DimValues& inputDims, const DimValues& outputDims,
              int kernelSizeX, int kernelSizeY,
              int kernelStride,
              int paddingX, int paddingY)
        : _stageName(stageName),
          _inputDims(inputDims), _outputDims(outputDims),
          _kernelSizeX(kernelSizeX), _kernelSizeY(kernelSizeY),
          _kernelStride(kernelStride),
          _paddingX(paddingX), _paddingY(paddingY) {
    }

    bool optimize() {
        initTileSizes();

        if (!selectBestTile()) {
            return false;
        }

        return createTiles();
    }

    const HwPoolTilingPtr& getTiling() const {
        return _tiling;
    }

private:
    void initTileSizes() {
        int tempX = _inputDims[Dim::W] + 2 * _paddingX - _kernelSizeX;
        int tempY = _inputDims[Dim::H] + 2 * _paddingY - _kernelSizeY;

        int outWidthWithOutCeil = (tempX + _kernelStride) / _kernelStride;
        int outHeightWithOutCeil = (tempY + _kernelStride) / _kernelStride;

        int outWidthWithCeil =  static_cast<int>(std::ceil(static_cast<double>(tempX) / _kernelStride + 1));
        int outHeightWithCeil = static_cast<int>(std::ceil(static_cast<double>(tempY) / _kernelStride + 1));

        if ((_outputDims[Dim::W] != outWidthWithCeil) && (_outputDims[Dim::W] != outWidthWithOutCeil)) {
            VPU_THROW_EXCEPTION
                    << "Internal error: Output in " << _stageName << " has incorrect width dimension. Expected: "
                    << outWidthWithCeil << " or " << outWidthWithOutCeil << " Actual: " << _outputDims[Dim::W];
        }

        if ((_outputDims[Dim::H] != outHeightWithCeil) && (_outputDims[Dim::H] != outHeightWithOutCeil)) {
            VPU_THROW_EXCEPTION
                    << "Internal error: Output in " << _stageName << " has incorrect height dimension. Expected: "
                    << outHeightWithCeil << " or " << outHeightWithOutCeil << " Actual: " << _outputDims[Dim::H];
        }

        if ((_outputDims[Dim::W] == outWidthWithCeil) && (_outputDims[Dim::H] == outHeightWithCeil)) {
            _useCeil = true;
        } else {
            IE_ASSERT((_outputDims[Dim::W] == outWidthWithOutCeil) && (_outputDims[Dim::H] == outHeightWithOutCeil));
        }

        _inputTileDims.set(Dim::W, _inputDims[Dim::W]);
        _inputTileDims.set(Dim::H, _inputDims[Dim::H]);
        _inputTileDims.set(Dim::C, _inputDims[Dim::C]);
        _inputTileDims.set(Dim::N, _inputDims.get(Dim::N, 1));

        _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
        _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
        _outputTileDims.set(Dim::C, _outputDims[Dim::C]);
        _outputTileDims.set(Dim::N, _outputDims.get(Dim::N, 1));
    }

    bool selectBestTile() {
        struct Solution final {
            int numWidthTiles = 0;
            int numHeightTiles = 0;
            int numBatchTiles = 0;
            int totalNumTiles = 0;
            double cost = std::numeric_limits<double>::max();
        };

        const auto& env = CompileEnv::get();

        // TODO: estimate this numbers
        const int maxNumWidthTiles = 15;
        const int maxNumHeightTiles = 15;
        const int maxNumBatchTiles = _outputDims.get(Dim::N, 1);

        Solution bestSol;

        auto outputTileCopy = _outputTileDims;

        for (int numBatchTiles = 1; numBatchTiles <= maxNumBatchTiles; numBatchTiles++) {
            //
            // Filter-out misaligned SoN tiles.
            //

            if (outputTileCopy[Dim::N] % numBatchTiles != 0) {
                continue;
            }

            auto tileDimN = outputTileCopy[Dim::N] / numBatchTiles;

            for (int numWidthTiles = 1; numWidthTiles <= maxNumWidthTiles; numWidthTiles++) {
                auto inputTileDimW = divUp(_inputDims[Dim::W], numWidthTiles);

                //
                // Filter-out too small SoW tiles.
                //

                if (numWidthTiles > 1 && (inputTileDimW < 8 || inputTileDimW < _kernelSizeX)) {
                    break;
                }

                for (int numHeightTiles = 1; numHeightTiles <= maxNumHeightTiles ; numHeightTiles++) {
                    auto inputTileDimH = divUp(_inputDims[Dim::H], numHeightTiles);

                    //
                    // Filter-out too small SoH tiles.
                    //

                    if (numHeightTiles > 1 && inputTileDimH < _kernelSizeY) {
                        break;
                    }

                    //
                    // Try current tile size.
                    //

                    _inputTileDims.set(Dim::W, inputTileDimW);
                    _inputTileDims.set(Dim::H, inputTileDimH);
                    _inputTileDims.set(Dim::N, tileDimN);

                    _outputTileDims = outputTileCopy;
                    _outputTileDims.set(Dim::N, tileDimN);
                    correctOutputPlaneSize();

                    //
                    // Check that tiling is valid.
                    //

                    auto heightTiles = calcHeightTiles();
                    auto widthTiles = calcWidthTiles();

                    if (heightTiles.empty()) {
                        continue;
                    }
                    if (widthTiles.empty()) {
                        break;
                    }

                    bool isOK = true;
                    double solutionCost = 0.0;

                    for (const auto& heightTile : heightTiles) {
                        for (const auto& widthTile : widthTiles) {
                            //
                            // Output tile fits to CMX limitation.
                            //

                            DimValues fullOutputTileDims;
                            fullOutputTileDims.set(Dim::W, widthTile.outputWithJunk);
                            fullOutputTileDims.set(Dim::H, heightTile.outputWithJunk);
                            fullOutputTileDims.set(Dim::C, _outputTileDims[Dim::C]);
                            fullOutputTileDims.set(Dim::N, _outputTileDims[Dim::N]);

                            // TODO: support HCW
                            if (calculateHwBufferSize(fullOutputTileDims) > env.resources.cmxLimit) {
                                isOK = false;
                                break;
                            }

                            //
                            // `linesPerChan` restrictions.
                            //

                            if (heightTile.inputWithJunk < _kernelSizeY) {
                                isOK = false;
                                break;
                            }

                            const uint32_t LOCAL_RAM_SIZE = 128 * 1024;
                            const uint32_t CMX_DATA_BIT_WIDTH = 128;

                            uint32_t sizeOfBlock = LOCAL_RAM_SIZE >> static_cast<uint32_t>(HwOpMode::MODE_16_16);
                            uint32_t bytesPerPixel = 1 << (1 - static_cast<uint32_t>(HwDataMode::FP16));
                            uint32_t pixelsPerCMXLine = CMX_DATA_BIT_WIDTH / (bytesPerPixel * 8u);
                            uint32_t localLineStride = (widthTile.inputWithJunk + (pixelsPerCMXLine - 1)) / pixelsPerCMXLine;
                            uint32_t chanPerBlock = 1;
                            uint32_t availableBytesPerChan = sizeOfBlock / chanPerBlock;
                            uint32_t bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel;
                            uint32_t linesPerChan = availableBytesPerChan / bytesPerLine;
                            if (linesPerChan < _kernelSizeY) {
                                isOK = false;
                                break;
                            }

                            //
                            // Replicate padding in case of large input plane - #-16783.
                            //

                            DimValues fullInputTileDims;
                            fullInputTileDims.set(Dim::W, widthTile.inputWithJunk);
                            fullInputTileDims.set(Dim::H, heightTile.inputWithJunk);

                            auto pad = getHwPaddingInfo(
                                fullInputTileDims, fullOutputTileDims,
                                _kernelSizeX, _kernelSizeY,
                                _kernelStride, _kernelStride);

                            if (pad.enable) {
                                int memPerPlane = alignVal(
                                            fullInputTileDims[Dim::W], 8) * sizeof(fp16_t)
                                          * ((fullInputTileDims[Dim::H] - 1) + (_kernelSizeY - 1));
                                int memLimit = pad.bottom > 0 ? 0x800 : 0x1000;
                                if (memPerPlane > memLimit) {
                                    isOK = false;
                                    break;
                                }
                            }

                            //
                            // Calc tile cost.
                            //

                            auto noOfBlocks = 1 << static_cast<int>(HwOpMode::MODE_16_16);
                            solutionCost += 1.0
                                  * ((_inputTileDims[Dim::C] * _inputTileDims[Dim::N]) / noOfBlocks) * _kernelSizeX * _kernelSizeY
                                  * numBatchTiles;

                            // Alignment for output
                            if ((widthTile.outputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                                solutionCost += 1.0
                                      * widthTile.outputWithJunk
                                      * heightTile.outputWithJunk
                                      * _outputTileDims[Dim::C]
                                      * _outputTileDims[Dim::N];
                            }

                            // Alignment for input
                            if ((widthTile.inputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                                solutionCost += 1.0
                                      * widthTile.inputWithJunk
                                      * heightTile.inputWithJunk
                                      * _inputTileDims[Dim::C]
                                      * _inputTileDims[Dim::N];
                            }
                        }

                        if (!isOK) {
                            break;
                        }
                    }

                    if (!isOK) {
                        continue;
                    }

                    //
                    // Compare with current best solution.
                    //

                    Solution curSol;
                    curSol.numWidthTiles = numWidthTiles;
                    curSol.numHeightTiles = numHeightTiles;
                    curSol.numBatchTiles = numBatchTiles;
                    curSol.totalNumTiles = numWidthTiles * numHeightTiles * numBatchTiles;
                    curSol.cost = solutionCost;

                    if (curSol.cost < bestSol.cost || (isDoubleEqual(curSol.cost, bestSol.cost) && curSol.totalNumTiles < bestSol.totalNumTiles)) {
                        bestSol = curSol;
                    }
                }
            }
        }

        if (bestSol.totalNumTiles == 0) {
            return false;
        }

        int inputTileDimW = divUp(_inputDims[Dim::W], bestSol.numWidthTiles);
        int inputTileDimH = divUp(_inputDims[Dim::H], bestSol.numHeightTiles);
        auto tileDimN = outputTileCopy[Dim::N] / bestSol.numBatchTiles;

        _inputTileDims.set(Dim::W, inputTileDimW);
        _inputTileDims.set(Dim::H, inputTileDimH);
        _inputTileDims.set(Dim::N, tileDimN);

        _outputTileDims = outputTileCopy;
        _outputTileDims.set(Dim::N, tileDimN);
        correctOutputPlaneSize();

        return true;
    }

    bool createTiles() {
        auto heightTiles = calcHeightTiles();
        IE_ASSERT(!heightTiles.empty());

        auto widthTiles = calcWidthTiles();
        IE_ASSERT(!widthTiles.empty());

        _tiling = std::make_shared<HwPoolTiling>();
        _tiling->sohTiles = heightTiles.size();
        _tiling->sowTiles = widthTiles.size();
        _tiling->socTiles = divUp(_inputDims.get(Dim::N, 1), _inputTileDims[Dim::N]);

        for (int sohInd = 0; sohInd < _tiling->sohTiles; ++sohInd) {
            const auto& heightTileInfo = heightTiles[sohInd];

            for (int sowInd = 0; sowInd < _tiling->sowTiles; ++sowInd) {
                const auto& widthTileInfo = widthTiles[sowInd];

                auto planeTile = std::make_shared<HwPoolPlaneTile>();
                planeTile->parent = _tiling;

                planeTile->sohInd = sohInd;
                planeTile->sowInd = sowInd;

                planeTile->heightInfo = heightTileInfo;
                planeTile->widthInfo = widthTileInfo;

                for (int socInd = 0; socInd < _tiling->socTiles; ++socInd) {
                    auto channelTile = std::make_shared<HwPoolChannelTile>();
                    channelTile->parent = planeTile;

                    channelTile->socInd = socInd;

                    channelTile->finalTiles = splitPooling(_inputTileDims[Dim::C] * _inputTileDims[Dim::N]);

                    if (channelTile->finalTiles.numDescr == 0) {
                        return false;
                    }

                    channelTile->channelStartIndex = socInd * _inputTileDims[Dim::N];
                    channelTile->numInputChannels = _inputTileDims[Dim::N];

                    planeTile->channelTiles.emplace_back(channelTile);
                }

                _tiling->planeTiles.emplace_back(planeTile);
            }
        }

        return true;
    }

private:
    void correctOutputPlaneSize() {
        int maxOutputWidth = calcOutputSize(_inputTileDims[Dim::W], _kernelSizeX, _kernelStride, _paddingX, _paddingX, _useCeil);
        _outputTileDims.set(Dim::W, std::min(_outputTileDims[Dim::W], maxOutputWidth));

        int maxOutputHeight = calcOutputSize(_inputTileDims[Dim::H], _kernelSizeY, _kernelStride, _paddingY, _paddingY, _useCeil);
        _outputTileDims.set(Dim::H, std::min(_outputTileDims[Dim::H], maxOutputHeight));
    }

    std::vector<HwPlaneTileInfo> calcHeightTiles() {
        std::vector<HwPlaneTileInfo> heightTiles;

        if (_outputTileDims[Dim::H] == _outputDims[Dim::H]) {
            HwPlaneTileInfo info;
            info.inputWithJunk = _inputDims[Dim::H];
            info.outputWithJunk = _outputDims[Dim::H];
            info.outputJunkBefore = 0;
            info.outputJunkAfter = 0;
            info.inputStartIndex = 0;
            info.inputEndIndex = _inputDims[Dim::H];
            info.outputStartIndex = 0;
            info.outputEndIndex = _outputDims[Dim::H];

            heightTiles.emplace_back(info);
        } else {
            heightTiles = splitIntoPlaneTiles(
                _inputDims[Dim::H],
                _outputDims[Dim::H],
                _kernelSizeY,
                _kernelStride,
                _paddingY, _paddingY,
                _outputTileDims[Dim::H],
                false,
                _useCeil);
        }

        return heightTiles;
    }

    std::vector<HwPlaneTileInfo> calcWidthTiles() {
        std::vector<HwPlaneTileInfo> widthTiles;

        if (_outputTileDims[Dim::W] == _outputDims[Dim::W]) {
            HwPlaneTileInfo info;
            info.inputWithJunk = _inputDims[Dim::W];
            info.outputWithJunk = _outputDims[Dim::W];
            info.outputJunkBefore = 0;
            info.outputJunkAfter = 0;
            info.inputStartIndex = 0;
            info.inputEndIndex = _inputDims[Dim::W];
            info.outputStartIndex = 0;
            info.outputEndIndex = _outputDims[Dim::W];

            widthTiles.emplace_back(info);
        } else {
            widthTiles = splitIntoPlaneTiles(
                _inputDims[Dim::W],
                _outputDims[Dim::W],
                _kernelSizeX,
                _kernelStride,
                _paddingX, _paddingX,
                _outputTileDims[Dim::W],
                true,
                _useCeil);
        }

        return widthTiles;
    }

private:
    std::string _stageName;

    DimValues _inputDims;
    DimValues _outputDims;

    int _kernelSizeX = 0;
    int _kernelSizeY = 0;
    int _kernelStride = 0;
    int _paddingX = 0;
    int _paddingY = 0;

    DimValues _inputTileDims;
    DimValues _outputTileDims;

    HwPoolTilingPtr _tiling;

    bool _useCeil = false;
};

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuidler) : _stageBuidler(stageBuidler) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuidler;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(hwPoolTiling);

    for (const auto& origStage : model->getStages()) {
        if (origStage->type() != StageType::StubMaxPool &&
            origStage->type() != StageType::StubAvgPool) {
            continue;
        }

        auto tryHW = origStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        auto origInput = origStage->input(0);
        auto origOutput = origStage->output(0);

        auto kernelSizeX = origStage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = origStage->attrs().get<int>("kernelSizeY");
        auto kernelStride = origStage->attrs().get<int>("kernelStrideX");
        auto padLeft = origStage->attrs().get<int>("padLeft");
        auto padRight = origStage->attrs().get<int>("padRight");
        auto padTop = origStage->attrs().get<int>("padTop");
        auto padBottom = origStage->attrs().get<int>("padBottom");

        auto withReLU = origStage->attrs().getOrDefault<bool>("withReLU", false);

        auto hwInput  = origInput;
        auto hwOutput = origOutput;

        //
        // Try to find "best" tiling
        //

        Optimizer opt(origStage->name(),
                      hwInput->desc().dims(), hwOutput->desc().dims(),
                      kernelSizeX, kernelSizeY,
                      kernelStride,
                      padLeft, padTop);

        if (!opt.optimize()) {
            origStage->attrs().set<bool>("tryHW", false);

            auto swOutput = origOutput;
            if (withReLU) {
                swOutput = model->addNewData(
                    origStage->name(),
                    origOutput->desc());
                swOutput->attrs().copyFrom(origOutput->attrs());

                model->replaceStageOutput(origStage->outputEdge(0), swOutput);

                _stageBuidler->addReLUStage(
                    model,
                    origStage->name() + "@ReLU",
                    origStage->origLayer(),
                    0.0,
                    swOutput,
                    origOutput);
            }

            continue;
        }

        //
        // Create HW tiles
        //

        model->disconnectStageDatas(origStage);

        const auto& tiling = opt.getTiling();

        DataVector hwInputTiles;
        std::vector<DimValues> hwInputTilesOffsets;

        DataVector hwOutputTiles;
        std::vector<DimValues> hwOutputTilesOffsets;

        for (const auto& planeTile : tiling->planeTiles) {
            for (const auto& channelTile : planeTile->channelTiles) {
                auto tilePostfix = getPlaneTilePostfix(planeTile) + getChannelTilePostfix(channelTile);

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
                    newDesc.setDim(Dim::N, channelTile->numInputChannels);

                    hwInputTile = model->duplicateData(
                        hwInput,
                        tilePostfix,
                        newDesc);

                    hwInputTiles.emplace_back(hwInputTile);
                    hwInputTilesOffsets.emplace_back(
                        DimValues({
                            {Dim::W, planeTile->widthInfo.inputStartIndex},
                            {Dim::H, planeTile->heightInfo.inputStartIndex},
                            {Dim::N, channelTile->channelStartIndex}
                        }));
                }

                //
                // Add alignement to input tile if needed
                //

                if ((planeTile->widthInfo.inputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                    auto hwInputTileAligned = model->duplicateData(
                        hwInputTile,
                        "@aligned");

                    _stageBuidler->addCopyStage(
                        model,
                        origStage->name() + tilePostfix + "@align-input-ptr",
                        origStage->origLayer(),
                        hwInputTile,
                        hwInputTileAligned);

                    hwInputTile = hwInputTileAligned;
                }

                //
                // Create output tile
                //

                Data hwOutputTile;

                if (tiling->sohTiles == 1 && tiling->sowTiles == 1 && tiling->socTiles == 1) {
                    hwOutputTile = hwOutput;
                } else {
                    auto newDesc = hwOutput->desc();
                    newDesc.setDim(Dim::W, planeTile->widthInfo.outputEndIndex - planeTile->widthInfo.outputStartIndex);
                    newDesc.setDim(Dim::H, planeTile->heightInfo.outputEndIndex - planeTile->heightInfo.outputStartIndex);
                    newDesc.setDim(Dim::N, channelTile->numInputChannels);

                    hwOutputTile = model->duplicateData(
                        hwOutput,
                        tilePostfix,
                        newDesc);

                    hwOutputTiles.emplace_back(hwOutputTile);
                    hwOutputTilesOffsets.emplace_back(
                        DimValues({
                            {Dim::W, planeTile->widthInfo.outputStartIndex},
                            {Dim::H, planeTile->heightInfo.outputStartIndex},
                            {Dim::N, channelTile->channelStartIndex}
                        }));
                }

                //
                // Add alignement to output tile if needed
                //

                if ((planeTile->widthInfo.outputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                    auto hwOutputTileAligned = model->duplicateData(
                        hwOutputTile,
                        "@aligned");

                    _stageBuidler->addCopyStage(
                        model,
                        origStage->name() + tilePostfix + "@align-output-ptr",
                        origStage->origLayer(),
                        hwOutputTileAligned,
                        hwOutputTile);

                    hwOutputTile = hwOutputTileAligned;
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

                    _stageBuidler->addShrinkStage(
                        model,
                        origStage->name() + tilePostfix + "@remove-junk",
                        origStage->origLayer(),
                        hwOutputTileWithJunk,
                        hwOutputTile,
                        innerOffset);

                    hwOutputTile = hwOutputTileWithJunk;
                }

                //
                // Create HW stage for tile
                //

                auto hwPad = getHwPaddingInfo(
                    hwInputTile->desc().dims(), hwOutputTile->desc().dims(),
                    kernelSizeX, kernelSizeY,
                    kernelStride, kernelStride);

                auto hwTileWeights = model->addFakeData();
                auto hwTileBiases = model->addFakeData();
                auto hwTileScales = model->addFakeData();

                auto hwStage = model->addNewStage<MyriadXHwStage>(
                    origStage->name() + tilePostfix,
                    StageType::MyriadXHwOp,
                    origStage->origLayer(),
                    {hwInputTile, hwTileWeights, hwTileBiases, hwTileScales},
                    {hwOutputTile});

                hwStage->attrs().set<HwOpType>("hwOpType", HwOpType::POOL);
                hwStage->attrs().set<HwPoolType>("poolType", origStage->type() == StageType::StubMaxPool ? HwPoolType::MAX : HwPoolType::AVERAGE);

                hwStage->attrs().set<int>("kernelSizeX", kernelSizeX);
                hwStage->attrs().set<int>("kernelSizeY", kernelSizeY);
                hwStage->attrs().set<int>("kernelStride", kernelStride);

                hwStage->attrs().set("pad", hwPad);

                hwStage->attrs().set<HwPoolTileInfo>("tiling", channelTile->finalTiles);

                hwStage->attrs().set<bool>("withReLU", withReLU);
            }
        }

        //
        // Split/concat input/output tiles
        //

        if (!hwInputTiles.empty()) {
            _stageBuidler->addSplitStage(
                model,
                origStage->name() + "@split-input",
                origStage->origLayer(),
                hwInputTilesOffsets,
                hwInput,
                hwInputTiles);
        }

        if (!hwOutputTiles.empty()) {
            _stageBuidler->addConcatStage(
                model,
                origStage->name() + "@concat-output",
                origStage->origLayer(),
                hwOutputTilesOffsets,
                hwOutputTiles,
                hwOutput);
        }

        //
        // Remove SW stage
        //

        model->removeStage(origStage);
    }
}

}  // namespace

Pass::Ptr PassManager::hwPoolTiling() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
