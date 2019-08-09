// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

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

#include <precision_utils.h>

#include <vpu/compile_env.hpp>
#include <vpu/stub_stage.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/tiling.hpp>
#include <vpu/hw/utility.hpp>

namespace vpu {

namespace {

class Optimizer final {
public:
    Optimizer(const std::string& stageName,
              const DimValues& inputDims, const DimValues& outputDims,
              const DimValues& origOutputDims,
              bool withPool,
              int kernelSizeX, int kernelSizeY,
              int kernelStride,
              int paddingX, int paddingY)
        : _stageName(stageName),
          _inputDims(inputDims), _outputDims(outputDims),
          _origOutputDims(origOutputDims),
          _withPool(withPool),
          _kernelSizeX(kernelSizeX), _kernelSizeY(kernelSizeY),
          _kernelStride(kernelStride),
          _paddingX(paddingX), _paddingY(paddingY) {
    }

    bool optimize() {
        initTileSizes();

        if (!selectBestTile()) {
            if (_withPool) {
                removePool();
                return optimize();
            }

            return false;
        }

        patternMatching();

        // Merged Pooling and SoC can't be used together.
        if (_withPool) {
            IE_ASSERT(!hasSoC());
        }

        if (!createTiles()) {
            if (_withPool) {
                removePool();
                return optimize();
            }

            return false;
        }

        return true;
    }

    bool withPool() const {
        return _withPool;
    }

    const HwConvTilingPtr& getTiling() const {
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

        if ((_origOutputDims[Dim::W] != outWidthWithCeil) && (_origOutputDims[Dim::W] != outWidthWithOutCeil)) {
            VPU_THROW_EXCEPTION
                    << "Internal error: Output in " << _stageName << " has incorrect width dimension. Expected: "
                    << outWidthWithCeil << " or " << outWidthWithOutCeil << " Actual: " << _origOutputDims[Dim::W];
        }

        if ((_origOutputDims[Dim::H] != outHeightWithCeil) && (_origOutputDims[Dim::H] != outHeightWithOutCeil)) {
            VPU_THROW_EXCEPTION
                    << "Internal error: Output in " << _stageName << " has incorrect height dimension. Expected: "
                    << outHeightWithCeil << " or " << outHeightWithOutCeil << " Actual: " << _origOutputDims[Dim::H];
        }

        if ((_origOutputDims[Dim::W] == outWidthWithCeil) && (_origOutputDims[Dim::H] == outHeightWithCeil)) {
            _useCeil = true;
        } else {
            IE_ASSERT((_origOutputDims[Dim::W] == outWidthWithOutCeil) && (_origOutputDims[Dim::H] == outHeightWithOutCeil));
        }

        _inputTileDims.set(Dim::W, std::min(CNN_MAX_INPUT_WIDTH, _inputDims[Dim::W]));
        _inputTileDims.set(Dim::H, std::min(CNN_MAX_INPUT_HEIGHT, _inputDims[Dim::H]));
        _inputTileDims.set(Dim::C, std::min(CNN_MAX_INPUT_CHANNELS, _inputDims[Dim::C]));

        _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
        _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
        _outputTileDims.set(Dim::C, _outputDims[Dim::C]);

        correctOutputPlaneSize();
    }

    void patternMatching() {
        if (!_withPool &&
            _kernelSizeX == 3 && _kernelSizeY == 3 && _paddingX == 1 && _paddingY == 1 && _kernelStride == 1 &&
            _inputDims[Dim::C] == 512 && _inputDims[Dim::H] == 28 && _inputDims[Dim::W] == 28 &&
            _outputDims[Dim::C] == 512) {
            _inputTileDims.set(Dim::H, 28);
            _inputTileDims.set(Dim::C, 172);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
            _kernelSizeX == 3 && _kernelSizeY == 3 && _paddingX == 1 && _paddingY == 1 && _kernelStride == 1 &&
            _inputDims[Dim::C] == 256 && _inputDims[Dim::H] == 56 && _inputDims[Dim::W] == 56 &&
            _outputDims[Dim::C] == 256) {
            _inputTileDims.set(Dim::H, 30);
            _inputTileDims.set(Dim::C, 128);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
            _kernelSizeX == 3 && _kernelSizeY == 3 && _paddingX == 1 && _paddingY == 1 && _kernelStride == 1 &&
            _inputDims[Dim::C] == 64 && _inputDims[Dim::H] == 224 && _inputDims[Dim::W] == 224 &&
            _outputDims[Dim::C] == 64) {
            _inputTileDims.set(Dim::H, 82);
            _inputTileDims.set(Dim::W, 82);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (_inputDims[Dim::C] == 512 &&
                _inputDims[Dim::H] == 7 &&
                _inputDims[Dim::W] == 7 &&
                _outputDims[Dim::C] == 4096) {
            _inputTileDims.set(Dim::C, 64);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
            _kernelSizeX == 3 && _kernelSizeY == 3 && _paddingX == 1 && _paddingY == 1 && _kernelStride == 1 &&
            _inputDims[Dim::C] == 128 && _inputDims[Dim::H] == 112 && _inputDims[Dim::W] == 112 &&
            _outputDims[Dim::C] == 128) {
            _inputTileDims.set(Dim::H, 32);
            _inputTileDims.set(Dim::W, 112);
            _inputTileDims.set(Dim::C, 32);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (_inputDims[Dim::C] == 1088 &&
            _inputDims[Dim::H] == 17 &&
            _inputDims[Dim::W] == 17 &&
            (_outputDims[Dim::C] == 128 || _outputDims[Dim::C] == 192)) {
            _inputTileDims.set(Dim::H, 17);
            _inputTileDims.set(Dim::C, 544);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (_inputDims[Dim::C] == 1024 &&
                _inputDims[Dim::H] == 17 &&
                _inputDims[Dim::W] == 17 &&
                _outputDims[Dim::C] == 384) {
            _inputTileDims.set(Dim::H, 17);
            _inputTileDims.set(Dim::C, 512);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
            _kernelSizeX == 3 && _kernelSizeY == 3 && _paddingX == 0 && _paddingY == 0 && _kernelStride == 2 &&
            _inputDims[Dim::C] == 384 && _inputDims[Dim::H] == 35 && _inputDims[Dim::W] == 35 &&
            _outputDims[Dim::C] == 384) {
            _inputTileDims.set(Dim::C, 194);
            _inputTileDims.set(Dim::H, 35);
            _inputTileDims.set(Dim::W, 35);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (_inputDims[Dim::C] == 192 &&
                _inputDims[Dim::H] == 71 &&
                _inputDims[Dim::W] == 71 &&
                _outputDims[Dim::H] == 35) {
            _inputTileDims.set(Dim::W, 71);
            _inputTileDims.set(Dim::C, 96);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
                _inputDims[Dim::C] == 256 &&
                _inputDims[Dim::H] == 128 &&
                _inputDims[Dim::W] == 128 &&
                _outputDims[Dim::C] == 256) {
            _inputTileDims.set(Dim::W, 128);
            _inputTileDims.set(Dim::H, 15);
            _inputTileDims.set(Dim::C, 64);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
                _inputDims[Dim::C] == 512 &&
                _inputDims[Dim::H] == 64 &&
                _inputDims[Dim::W] == 64 &&
                _outputDims[Dim::C] == 512) {
            _inputTileDims.set(Dim::W, 64);
            _inputTileDims.set(Dim::H, 10);
            _inputTileDims.set(Dim::C, 128);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
            _kernelSizeX == 1 && _kernelSizeY == 1 && _paddingX == 0 && _paddingY == 0 && _kernelStride == 1 &&
            _inputDims[Dim::C] == 384 &&
            _inputDims[Dim::H] == 56 &&
            _inputDims[Dim::W] == 56 &&
            _outputDims[Dim::C] == 64) {
            _inputTileDims.set(Dim::C, 384);
            _inputTileDims.set(Dim::H, 56);
            _inputTileDims.set(Dim::W, 20);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
            _kernelSizeX == 1 && _kernelSizeY == 1 && _paddingX == 0 && _paddingY == 0 && _kernelStride == 1 &&
            _inputDims[Dim::C] == 2112 &&
            _inputDims[Dim::H] == 14 &&
            _inputDims[Dim::W] == 14 &&
            _outputDims[Dim::C] == 1056) {
            _inputTileDims.set(Dim::C, 556);
            _inputTileDims.set(Dim::H, 14);
            _inputTileDims.set(Dim::W, 14);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
            _kernelSizeX == 3 && _kernelSizeY == 3 && _paddingX == 1 && _paddingY == 1 && _kernelStride == 2 &&
            _inputDims[Dim::C] == 256 &&
            _inputDims[Dim::H] == 52 &&
            _inputDims[Dim::W] == 52 &&
            _outputDims[Dim::C] == 512) {
            _inputTileDims.set(Dim::C, 128);
            _inputTileDims.set(Dim::H, 52);
            _inputTileDims.set(Dim::W, 52);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }

        if (!_withPool &&
            _kernelSizeX == 3 && _kernelSizeY == 3 && _paddingX == 1 && _paddingY == 1 && _kernelStride == 1 &&
            _inputDims[Dim::C] == 256 &&
            _inputDims[Dim::H] == 23 &&
            _inputDims[Dim::W] == 23 &&
            _outputDims[Dim::C] == 640) {
            _inputTileDims.set(Dim::C, 256);
            _inputTileDims.set(Dim::H, 14);
            _inputTileDims.set(Dim::W, 23);
            _outputTileDims.set(Dim::H, _outputDims[Dim::H]);
            _outputTileDims.set(Dim::W, _outputDims[Dim::W]);
            correctOutputPlaneSize();
            return;
        }
    }

    bool selectBestTile() {
        struct Solution final {
            int numWidthTiles = 0;
            int numHeightTiles = 0;
            int numChannelTiles = 0;
            int totalNumTiles = 0;
            double cost = std::numeric_limits<double>::max();
        };

        const auto& env = CompileEnv::get();

        // TODO: estimate this numbers
        const int maxNumWidthTiles = 15;
        const int maxNumHeightTiles = 15;
        const int maxNumChannelTiles = _withPool ? 1 : 15;

        Solution bestSol;

        auto outputTileCopy = _outputTileDims;

        auto minInputTileDimW = 64;
        auto minInputTileDimH = _kernelSizeY;
        if (_withPool) {
            minInputTileDimW *= 2;
            minInputTileDimH *= 2;
        }

        for (int numChannelTiles = 1; numChannelTiles <= maxNumChannelTiles; numChannelTiles++) {
            int inputTileDimC = divUp(_inputDims[Dim::C], numChannelTiles);

            for (int numWidthTiles = 1; numWidthTiles <= maxNumWidthTiles; numWidthTiles++) {
                int inputTileDimW = divUp(_inputDims[Dim::W], numWidthTiles);

                //
                // Filter-out too small SoW tiles.
                //

                if (numWidthTiles > 1 && inputTileDimW < minInputTileDimW) {
                    break;
                }

                for (int numHeightTiles = 1; numHeightTiles <= maxNumHeightTiles; numHeightTiles++) {
                    int inputTileDimH = divUp(_inputDims[Dim::H], numHeightTiles);

                    //
                    // Filter-out too small SoH tiles.
                    //

                    if (numHeightTiles > 1 && inputTileDimH < minInputTileDimH) {
                        break;
                    }

                    //
                    // Try current tile size.
                    //

                    _inputTileDims.set(Dim::W, inputTileDimW);
                    _inputTileDims.set(Dim::H, inputTileDimH);
                    _inputTileDims.set(Dim::C, inputTileDimC);

                    _outputTileDims = outputTileCopy;
                    correctOutputPlaneSize();

                    //
                    // Limitations for Conv+Pool case.
                    //

                    if (_withPool) {
                        if (_outputTileDims[Dim::W] <= 2 ||
                            _outputTileDims[Dim::H] <= 2) {
                            break;
                        }
                    }

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
                            // Limitations for Conv+Pool case.
                            //

                            if (_withPool) {
                                if (widthTile.inputWithJunk % 2 != 0 ||
                                    heightTile.inputWithJunk % 2 != 0 ||
                                    widthTile.outputWithJunk % 2 != 0 ||
                                    widthTile.outputWithJunk <= 2 ||
                                    heightTile.outputWithJunk <= 2) {
                                    isOK = false;
                                    break;
                                }
                            }

                            //
                            // Can use this tile.
                            //

                            auto tileInfo = splitHwConvIntoOutChannelsTiles(
                                widthTile.inputWithJunk, heightTile.inputWithJunk, inputTileDimC,
                                outputTileCopy[Dim::C],
                                _kernelSizeX, _kernelSizeY, _kernelStride);

                            if (tileInfo.numDescr == 0) {
                                isOK = false;
                                break;
                            }

                            //
                            // Output tile fits to CMX limitation.
                            //

                            DimValues fullOutputTileDims;
                            fullOutputTileDims.set(Dim::W, widthTile.outputWithJunk);
                            fullOutputTileDims.set(Dim::H, heightTile.outputWithJunk);
                            fullOutputTileDims.set(Dim::C, outputTileCopy[Dim::C]);

                            // TODO: support HCW
                            if (calculateHwBufferSize(fullOutputTileDims) > env.resources.cmxLimit) {
                                isOK = false;
                                break;
                            }

                            //
                            // Calc tile cost.
                            //

                            solutionCost += tileInfo.cost * numChannelTiles;

                            // Alignment for output
                            if ((widthTile.outputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                                solutionCost += 1.0
                                      * widthTile.outputWithJunk
                                      * heightTile.outputWithJunk
                                      * outputTileCopy[Dim::C];
                            }

                            // Alignment for input
                            if ((widthTile.inputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                                solutionCost += 1.0
                                      * widthTile.inputWithJunk
                                      * heightTile.inputWithJunk
                                      * tileInfo.extendedInputDimC;
                            }

                            // SoC overhead
                            solutionCost += 1.0
                                  * (numChannelTiles - 1)
                                  * widthTile.outputWithJunk
                                  * heightTile.outputWithJunk
                                  * outputTileCopy[Dim::C];
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
                    curSol.numChannelTiles = numChannelTiles;
                    curSol.totalNumTiles = numWidthTiles * numHeightTiles * numChannelTiles;
                    curSol.cost = solutionCost;

                    if (curSol.cost < bestSol.cost || (isDoubleEqual(curSol.cost, bestSol.cost) && curSol.totalNumTiles < bestSol.totalNumTiles)) {
                        bestSol = curSol;
                    }

                    // Skip smaller SoC tiling.
                    break;
                }
            }
        }

        if (bestSol.totalNumTiles == 0) {
            return false;
        }

        int inputTileDimW = divUp(_inputDims[Dim::W], bestSol.numWidthTiles);
        int inputTileDimH = divUp(_inputDims[Dim::H], bestSol.numHeightTiles);
        int inputTileDimC = divUp(_inputDims[Dim::C], bestSol.numChannelTiles);

        _inputTileDims.set(Dim::W, inputTileDimW);
        _inputTileDims.set(Dim::H, inputTileDimH);
        _inputTileDims.set(Dim::C, inputTileDimC);

        _outputTileDims = outputTileCopy;
        correctOutputPlaneSize();

        return true;
    }

    bool createTiles() {
        auto heightTiles = calcHeightTiles();
        IE_ASSERT(!heightTiles.empty());

        auto widthTiles = calcWidthTiles();
        IE_ASSERT(!widthTiles.empty());

        _tiling = std::make_shared<HwConvTiling>();
        _tiling->sohTiles = heightTiles.size();
        _tiling->sowTiles = widthTiles.size();
        _tiling->socTiles = divUp(_inputDims[Dim::C], _inputTileDims[Dim::C]);

        for (int sohInd = 0; sohInd < _tiling->sohTiles; ++sohInd) {
            const auto& heightTileInfo = heightTiles[sohInd];

            for (int sowInd = 0; sowInd < _tiling->sowTiles; ++sowInd) {
                const auto& widthTileInfo = widthTiles[sowInd];

                auto planeTile = std::make_shared<HwConvPlaneTile>();
                planeTile->parent = _tiling;

                planeTile->sohInd = sohInd;
                planeTile->sowInd = sowInd;

                planeTile->heightInfo = heightTileInfo;
                planeTile->widthInfo = widthTileInfo;

                for (int socInd = 0; socInd < _tiling->socTiles; ++socInd) {
                    auto channelTile = std::make_shared<HwConvChannelTile>();
                    channelTile->parent = planeTile;

                    channelTile->socInd = socInd;

                    channelTile->finalTiles = splitHwConvIntoOutChannelsTiles(
                            widthTileInfo.inputWithJunk, heightTileInfo.inputWithJunk, _inputTileDims[Dim::C],
                            _outputTileDims[Dim::C],
                            _kernelSizeX, _kernelSizeY, _kernelStride);

                    if (channelTile->finalTiles.numDescr == 0) {
                        return false;
                    }

                    channelTile->extendedInputDimC = channelTile->finalTiles.extendedInputDimC;
                    channelTile->extendedOutputDimC = channelTile->finalTiles.extendedOutputDimC;

                    channelTile->channelStartIndex = socInd * _inputTileDims[Dim::C];
                    channelTile->numInputChannels = _inputTileDims[Dim::C];

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
        if (_withPool) {
            maxOutputWidth /= 2;
        }
        _outputTileDims.set(Dim::W, std::min(_outputTileDims[Dim::W], maxOutputWidth));

        int maxOutputHeight = calcOutputSize(_inputTileDims[Dim::H], _kernelSizeY, _kernelStride, _paddingY, _paddingY, _useCeil);
        if (_withPool) {
            maxOutputHeight /= 2;
        }
        _outputTileDims.set(Dim::H, std::min(_outputTileDims[Dim::H], maxOutputHeight));
    }

    bool hasSoC() const {
        return _inputTileDims[Dim::C] != _inputDims[Dim::C];
    }

    void removePool() {
        _withPool = false;
        _outputDims = _origOutputDims;
    }

    SmallVector<HwPlaneTileInfo> calcHeightTiles() {
        SmallVector<HwPlaneTileInfo> heightTiles;

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
            if (_withPool) {
                heightTiles = splitIntoPlaneTilesWithPool(
                    _inputDims[Dim::H],
                    _kernelSizeY,
                    _kernelStride,
                    _paddingY,
                    _outputTileDims[Dim::H]);
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
        }

        return heightTiles;
    }

    SmallVector<HwPlaneTileInfo> calcWidthTiles() {
        SmallVector<HwPlaneTileInfo> widthTiles;

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
            if (_withPool) {
                widthTiles = splitIntoPlaneTilesWithPool(
                    _inputDims[Dim::W],
                    _kernelSizeX,
                    _kernelStride,
                    _paddingX,
                    _outputTileDims[Dim::W]);
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
        }

        return widthTiles;
    }

private:
    std::string _stageName;

    DimValues _inputDims;
    DimValues _outputDims;
    DimValues _origOutputDims;

    bool _withPool = false;

    int _kernelSizeX = 0;
    int _kernelSizeY = 0;
    int _kernelStride = 0;
    int _paddingX = 0;
    int _paddingY = 0;

    DimValues _inputTileDims;
    DimValues _outputTileDims;

    HwConvTilingPtr _tiling;

    bool _useCeil = false;
};

using TileWeightsMap = std::unordered_map<int, Data>;

const int BIASES_IND = -1;
const int SCALES_IND = -2;

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(hwConvTiling);

    for (const auto& origStage : model->getStages()) {
        if (origStage->type() != StageType::StubConv) {
            continue;
        }

        auto tryHW = origStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        auto origInput = origStage->input(0);
        auto origWeights = origStage->input(1);
        auto origBiases = origStage->input(2);
        auto origOutput = origStage->output(0);

        auto kernelSizeX = origStage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = origStage->attrs().get<int>("kernelSizeY");
        auto kernelStride = origStage->attrs().get<int>("kernelStrideX");
        auto padLeft = origStage->attrs().get<int>("padLeft");
        auto padTop = origStage->attrs().get<int>("padTop");

        auto withReLU = origStage->attrs().getOrDefault<bool>("withReLU", false);
        auto negativeSlope = origStage->attrs().getOrDefault<float>("negativeSlope", 0.0f);
        auto a0 = origStage->attrs().getOrDefault<uint32_t>("a0", 0);
        auto a1 = origStage->attrs().getOrDefault<uint32_t>("a1", 0);
        auto reluScale = origStage->attrs().getOrDefault<float>("reluScale", 1.0f);

        auto withClamp = origStage->attrs().getOrDefault<bool>("withClamp", false);
        auto clampMax =  origStage->attrs().getOrDefault<float>("clampMax", 6.0);

        auto withPool = origStage->attrs().getOrDefault<bool>("withPool", false);
        auto poolKernelSizeX = origStage->attrs().getOrDefault<int>("poolKernelSizeX", 0);
        auto poolKernelSizeY = origStage->attrs().getOrDefault<int>("poolKernelSizeY", 0);
        auto poolKernelStride = origStage->attrs().getOrDefault<int>("poolKernelStride", 0);
        auto poolPadLeft = origStage->attrs().getOrDefault<int>("poolPadLeft", 0);
        auto poolPadRight = origStage->attrs().getOrDefault<int>("poolPadRight", 0);
        auto poolPadTop = origStage->attrs().getOrDefault<int>("poolPadTop", 0);
        auto poolPadBottom = origStage->attrs().getOrDefault<int>("poolPadBottom", 0);

        auto origOutputDesc = origStage->attrs().getOrDefault<DataDesc>("origConvOutput", origOutput->desc());

        auto scaleFactor = origStage->attrs().getOrDefault<float>("scaleFactor", 1.0f);

        auto& tileWeightsMap = origWeights->attrs().getOrSet<TileWeightsMap>("weightsPerTile", TileWeightsMap());

        //
        // Unsupported paddings
        //

        auto hwInput = origInput;
        auto hwOutput = origOutput;

        //
        // Try to find "best" tiling
        //

        Optimizer opt(origStage->name(),
                      hwInput->desc().dims(), hwOutput->desc().dims(),
                      origOutputDesc.dims(),
                      withPool,
                      kernelSizeX, kernelSizeY,
                      kernelStride,
                      padLeft, padTop);

        //
        // Use SW stage if tiling optimization failed
        //

        if (!opt.optimize()) {
            origStage->attrs().set<bool>("tryHW", false);

            auto swConvOutput = origOutput;
            if (withReLU || withPool || withClamp) {
                swConvOutput = model->addNewData(
                    origStage->name(),
                    origOutputDesc);
                swConvOutput->attrs().copyFrom(origOutput->attrs());

                model->replaceStageOutput(origStage->outputEdge(0), swConvOutput);
            }

            auto hwPoolInput = swConvOutput;
            if (withReLU) {
                auto swReluOutput = origOutput;
                if (withPool) {
                    swReluOutput = model->addNewData(
                        origStage->name() + "@ReLU",
                        origOutputDesc);
                    swReluOutput->attrs().copyFrom(origOutput->attrs());
                }

                _stageBuilder->addReLUStage(
                    model,
                    origStage->name() + "@ReLU",
                    origStage->origLayer(),
                    negativeSlope,
                    swConvOutput,
                    swReluOutput);

                hwPoolInput = swReluOutput;
            }

            if (withClamp) {
                auto swClampOutput = origOutput;
                if (withPool) {
                    swClampOutput = model->addNewData(
                            origStage->name() + "@Clamp",
                            origOutputDesc);
                    swClampOutput->attrs().copyFrom(origOutput->attrs());
                }

                _stageBuilder->addClampStage(
                        model,
                        origStage->name() + "@Clamp",
                        origStage->origLayer(),
                        0.0,
                        clampMax,
                        swConvOutput,
                        swClampOutput);

                hwPoolInput = swClampOutput;
            }

            if (withPool) {
                auto hwPoolStage = model->addNewStage<StubStage>(
                    origStage->name() + "@Pool",
                    StageType::StubMaxPool,
                    origStage->origLayer(),
                    {hwPoolInput},
                    {origOutput});

                hwPoolStage->attrs().set<int>("kernelSizeX", poolKernelSizeX);
                hwPoolStage->attrs().set<int>("kernelSizeY", poolKernelSizeY);

                hwPoolStage->attrs().set<int>("kernelStrideX", poolKernelStride);
                hwPoolStage->attrs().set<int>("kernelStrideY", poolKernelStride);

                hwPoolStage->attrs().set<int>("padLeft", poolPadLeft);
                hwPoolStage->attrs().set<int>("padRight", poolPadRight);
                hwPoolStage->attrs().set<int>("padTop", poolPadTop);
                hwPoolStage->attrs().set<int>("padBottom", poolPadBottom);

                hwPoolStage->attrs().set<bool>("excludePad", false);

                hwPoolStage->attrs().set<bool>("tryHW", true);
            }

            continue;
        }

        //
        // Remove merged pool if we failed to optimize tiling with it
        //

        model->disconnectStageDatas(origStage);

        if (withPool && !opt.withPool()) {
            auto hwPoolInput = model->addNewData(
                origStage->name(),
                origOutputDesc);
            hwPoolInput->attrs().copyFrom(origOutput->attrs());

            auto hwPoolStage = model->addNewStage<StubStage>(
                origStage->name() + "@Pool",
                StageType::StubMaxPool,
                origStage->origLayer(),
                {hwPoolInput},
                {hwOutput});

            hwPoolStage->attrs().set<int>("kernelSizeX", poolKernelSizeX);
            hwPoolStage->attrs().set<int>("kernelSizeY", poolKernelSizeY);

            hwPoolStage->attrs().set<int>("kernelStrideX", poolKernelStride);
            hwPoolStage->attrs().set<int>("kernelStrideY", poolKernelStride);

            hwPoolStage->attrs().set<int>("padLeft", poolPadLeft);
            hwPoolStage->attrs().set<int>("padRight", poolPadRight);
            hwPoolStage->attrs().set<int>("padTop", poolPadTop);
            hwPoolStage->attrs().set<int>("padBottom", poolPadBottom);

            hwPoolStage->attrs().set<bool>("excludePad", false);

            hwPoolStage->attrs().set<bool>("tryHW", true);

            hwOutput = hwPoolInput;

            withPool = false;
        }

        //
        // Broadcast input/output if needed
        //

        const auto& tiling = opt.getTiling();

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

            _stageBuilder->addBroadcastStage(
                model,
                origStage->name() + "@broadcast-input",
                origStage->origLayer(),
                hwInput,
                hwInputExtended);

            hwInput = hwInputExtended;
        }

        //
        // Create HW biases
        //

        auto hwBiases = tileWeightsMap[BIASES_IND];
        if (hwBiases == nullptr) {
            if (origBiases->usage() == DataUsage::Fake) {
                hwBiases = model->addFakeData();
            } else {
                auto origBiasesContent = origBiases->content();
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
                    origBiases,
                    "@HW",
                    DataDesc({maxExtendedOutputDimC}),
                    ieBlobContent(hwTileBiasesBlob));

                if (scaleFactor != 1.0f) {
                    auto hwBiasesScaled = model->duplicateData(
                        hwBiases,
                        formatString("@SCALE=%f", scaleFactor),
                        hwBiases->desc(),
                        scaleContent(hwBiases->content(), scaleFactor));
                    hwBiasesScaled->attrs().getOrSet<float>("scaleFactor", 1.0f) *= scaleFactor;

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
            float fullScale = 1.0f / scaleFactor;
            if (tiling->socTiles == 1 && reluScale != 1.0f) {
                fullScale *= reluScale;
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

        DataVector hwInputTiles;
        std::vector<DimValues> hwInputTilesOffsets;

        DataVector hwOutputTiles;
        std::vector<DimValues> hwOutputTilesOffsets;

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
                    hwOutputPlaneTile);

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
                        hwInputTileAligned);

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
                        if (channelTile->socInd < tiling->socTiles - 1 || withReLU || withClamp) {
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

                        if (channelTile->socInd == tiling->socTiles - 1 && withReLU) {
                            _stageBuilder->addReLUStage(
                                model,
                                origStage->name() + tilePostfix + "@ReLU",
                                origStage->origLayer(),
                                negativeSlope,
                                sumPartialOutput,
                                hwOutputTile);
                        }

                        if (channelTile->socInd == tiling->socTiles - 1 && withClamp) {
                            _stageBuilder->addClampStage(
                                    model,
                                    origStage->name() + tilePostfix + "@Clamp",
                                    origStage->origLayer(),
                                    0.0,
                                    clampMax,
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
                        origWeights,
                        "@HW" + channelTilePostfix,
                        DataDesc({8, kernelSizeX * kernelSizeY, channelTile->extendedInputDimC, channelTile->extendedOutputDimC / 8}),
                        std::make_shared<HwWeightsContent>(
                            origWeights->content(),
                            origWeights->desc(),
                            channelTile->numInputChannels,
                            channelTile->channelStartIndex));

                    if (scaleFactor != 1.0f) {
                        auto hwTileWeightsScaled = model->duplicateData(
                            hwTileWeights,
                            formatString("@SCALE=%f", scaleFactor),
                            hwTileWeights->desc(),
                            scaleContent(hwTileWeights->content(), scaleFactor));
                        hwTileWeightsScaled->attrs().getOrSet<float>("scaleFactor", 1.0f) *= scaleFactor;

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
                if (withPool) {
                    hwOutputTileDims.set(Dim::W, hwOutputTileDims[Dim::W] * poolKernelStride - poolPadLeft - poolPadRight);
                    hwOutputTileDims.set(Dim::H, hwOutputTileDims[Dim::H] * poolKernelStride - poolPadTop - poolPadBottom);
                }

                auto hwPad = getHwPaddingInfo(
                    hwInputTile->desc().dims(), hwOutputTileDims,
                    kernelSizeX, kernelSizeY,
                    kernelStride, kernelStride,
                    padLeft, padTop);

                auto hwStage = model->addNewStage<MyriadXHwStage>(
                    origStage->name() + tilePostfix,
                    StageType::MyriadXHwOp,
                    origStage->origLayer(),
                    {hwInputTile, hwTileWeights, hwTileBiases, hwScales},
                    {hwOutputTile});

                hwStage->attrs().set<HwOpType>("hwOpType", withPool ? HwOpType::CONV_POOL : HwOpType::CONV);

                hwStage->attrs().set<int>("kernelSizeX", kernelSizeX);
                hwStage->attrs().set<int>("kernelSizeY", kernelSizeY);
                hwStage->attrs().set<int>("kernelStride", kernelStride);

                if (withPool) {
                    hwStage->attrs().set<int>("poolKernelSizeX", poolKernelSizeX);
                    hwStage->attrs().set<int>("poolKernelSizeY", poolKernelSizeY);
                }

                hwStage->attrs().set<HwPaddingInfo>("pad", hwPad);

                hwStage->attrs().set<HwConvTileInfo>("tiling", channelTile->finalTiles);

                if (tiling->socTiles > 1) {
                    hwStage->attrs().set<bool>("withReLU", false);
                    hwStage->attrs().set<bool>("withClamp", false);
                } else {
                    hwStage->attrs().set<bool>("withReLU", withReLU);
                    hwStage->attrs().set<uint32_t>("a0", a0);
                    hwStage->attrs().set<uint32_t>("a1", a1);
                    hwStage->attrs().set<float>("negativeSlope", negativeSlope);

                    hwStage->attrs().set<bool>("withClamp", withClamp);
                    hwStage->attrs().set<float>("clampMax", clampMax);
                }

                hwStage->attrs().set<float>("scaleFactor", scaleFactor);
            }
        }

        //
        // Split/concat input/output tiles
        //

        if (!hwInputTiles.empty()) {
            _stageBuilder->addSplitStage(
                model,
                origStage->name() + "@split-input",
                origStage->origLayer(),
                std::move(hwInputTilesOffsets),
                hwInput,
                hwInputTiles);
        }

        if (!hwOutputTiles.empty()) {
            _stageBuilder->addConcatStage(
                model,
                origStage->name() + "@concat-output",
                origStage->origLayer(),
                std::move(hwOutputTilesOffsets),
                hwOutputTiles,
                hwOutput);
        }

        //
        // Remove original stage
        //

        model->removeStage(origStage);
    }
}

}  // namespace

Pass::Ptr PassManager::hwConvTiling() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
