// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <utility>
#include <vpu/passes/hw_pooling_tiling/hw_pooling_tiler.hpp>

namespace vpu {

namespace HWTilingNS {

class PoolingInputToOutputDirection;
class PoolingOutputToInputDirection;

// Input -> Output case
class PoolingInputToOutputDirection: public GraphDataTiling {
public:
    explicit PoolingInputToOutputDirection(const ConvolutionOptions &co): GraphDataTiling(co, Direction::INPUT_TO_OUTPUT) {}
    PoolingInputToOutputDirection(const PoolingInputToOutputDirection &other): GraphDataTiling(other) {}
    // ok
    void initTileSizes() override {
        _useCeil = ceilNeeded();

        _inputTileDims.set(Dim::W, _co._inputDims[Dim::W]);
        _inputTileDims.set(Dim::H, _co._inputDims[Dim::H]);
        _inputTileDims.set(Dim::C, _co._inputDims[Dim::C]);
        _inputTileDims.set(Dim::N, _co._inputDims.get(Dim::N, 1));

        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::C, _co._outputDims[Dim::C]);
        _outputTileDims.set(Dim::N, _co._outputDims.get(Dim::N, 1));
    }

    // Input -> Output case
    // ok
    void setInputNOutputTileDimensions(const int tileDimW, const int tileDimH, const int tileDimN) override {
        _inputTileDims.set(Dim::W, tileDimW);
        _inputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::N, tileDimN);

        _outputTileDims.set(Dim::N, tileDimN);

        correctOutputPlaneSize();
    }

    // Input -> Output case
    // ..
    void applyTilingOption(const TilingOption &tilingOption) override {
        int tileDimW = divUp(_co._inputDims[Dim::W], tilingOption.numWidthTiles);
        int tileDimH = divUp(_co._inputDims[Dim::H], tilingOption.numHeightTiles);
        const int tileDimN = divUp(_co._inputDims[Dim::N], tilingOption.numChannelTiles);

        tileDimW = divUp(tileDimW, _co._kernelStride) * _co._kernelStride;
        tileDimH = divUp(tileDimH, _co._kernelStride) * _co._kernelStride;

        _inputTileDims.set(Dim::W, tileDimW);
        _inputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::N, tileDimN);

        correctOutputPlaneSize();
    }

    void correctPlaneSize() override {
        correctOutputPlaneSize();
    }

    void correctOutputPlaneSize() {
        int maxOutputWidth = calcOutputSize(_inputTileDims[Dim::W], _co._kernelSizeX, _co._kernelStride, _co._paddingLeft, _co._paddingRight, _useCeil);
        _outputTileDims.set(Dim::W, std::min(_outputTileDims[Dim::W], maxOutputWidth));

        int maxOutputHeight = calcOutputSize(_inputTileDims[Dim::H], _co._kernelSizeY, _co._kernelStride, _co._paddingTop, _co._paddingBottom, _useCeil);
        _outputTileDims.set(Dim::H, std::min(_outputTileDims[Dim::H], maxOutputHeight));
    }

    const DimValues &splitOverTensorDims() override {
        return _co._inputDims;
    }

    void patternMatching() override {};

private:
    // ok
    bool ceilNeeded() {
        int tempX = _co._inputDims[Dim::W] + _co._paddingLeft + _co._paddingRight  - _co._kernelSizeX;
        int tempY = _co._inputDims[Dim::H] + _co._paddingTop  + _co._paddingBottom - _co._kernelSizeY;

        int outWidthWithOutCeil  = (tempX + _co._kernelStride) / _co._kernelStride;
        int outHeightWithOutCeil = (tempY + _co._kernelStride) / _co._kernelStride;

        int outWidthWithCeil =  static_cast<int>(std::ceil(static_cast<double>(tempX) / _co._kernelStride + 1));
        int outHeightWithCeil = static_cast<int>(std::ceil(static_cast<double>(tempY) / _co._kernelStride + 1));

        if ((_co._outputDims[Dim::W] != outWidthWithCeil) && (_co._outputDims[Dim::W] != outWidthWithOutCeil)) {
            VPU_THROW_EXCEPTION
                    << "Internal error: Output in " << _co._stageName << " has incorrect width dimension. Expected: "
                    << outWidthWithCeil << " or " << outWidthWithOutCeil << " Actual: " << _co._outputDims[Dim::W];
        }

        if ((_co._outputDims[Dim::H] != outHeightWithCeil) && (_co._outputDims[Dim::H] != outHeightWithOutCeil)) {
            VPU_THROW_EXCEPTION
                    << "Internal error: Output in " << _co._stageName << " has incorrect height dimension. Expected: "
                    << outHeightWithCeil << " or " << outHeightWithOutCeil << " Actual: " << _co._outputDims[Dim::H];
        }

        if ((_co._origOutputDims[Dim::W] == outWidthWithOutCeil) && (_co._origOutputDims[Dim::H] == outHeightWithOutCeil)) {
            return false;
        } else {
            return true;
        }
    }
};

HWPoolingTiler::HWPoolingTiler(const ConvolutionOptions &co,
                                   Direction direction,
                                   size_t maxTilingOptions) :
    _co(co),
    _searcher(_co, direction, maxTilingOptions) {
    _tilingPossible = tileForHW();
}

bool HWPoolingTiler::tileForHW() {
    const std::vector<TilingOption> &tilingOptions = _searcher.tilingOptions();
    if (tilingOptions.empty()) {
        return false;
    }

    for (const TilingOption &tilingOption : tilingOptions) {
        const HWPoolingTileLayoutCut tileLayoutCut = _searcher.tileLayoutCut(tilingOption);
        if (tileLayoutCut.tileCutPossible()) {
            _hwTilings.push_back(tileLayoutCut.hwTiling());
        }
    }

    return _hwTilings.size() != 0;
}

std::unique_ptr<GraphDataTiling> PoolGraphDataTilingFactory::makeDirTiling(const ConvolutionOptions &co,
                                                                           Direction direction) {
    if (direction == Direction::INPUT_TO_OUTPUT) {
        return std::unique_ptr<GraphDataTiling>(new PoolingInputToOutputDirection(co));
    // } else if (direction == Direction::OUTPUT_TO_INPUT) {
    //     return std::unique_ptr<GraphDataTiling>(new PoolingOutputToInputDirection(co));
    } else {
        IE_ASSERT(false) << "Unsupported direction";
    }
}

std::unique_ptr<GraphDataTiling> PoolGraphDataTilingFactory::makeDirTiling(const GraphDataTiling &o) {
    if (o.getDirection() == Direction::INPUT_TO_OUTPUT) {
        return std::unique_ptr<GraphDataTiling>(
                new PoolingInputToOutputDirection(dynamic_cast<const PoolingInputToOutputDirection&>(o)));
    // } else if (o.getDirection() == Direction::OUTPUT_TO_INPUT) {
    //     return std::unique_ptr<GraphDataTiling>(
    //             new PoolingOutputToInputDirection(dynamic_cast<const PoolingOutputToInputDirection&>(o)));
    } else {
        IE_ASSERT(false) << "Unsupported direction";
    }
}

//
// Looks for the optimal tiling accordingly to the cost function. Modifies dimensions in dirTiling during search.
//
std::vector<TilingOption> HWPoolingTilingSearcher::selectBetterTiling() const {
    const auto& env = CompileEnv::get();
    GraphDataTiling &dirTiling = *_dirTiling;
    FixedMaxHeap<TilingOption> tilingOptions(_maxTilingOptions);

    // TODO: estimate this numbers
    const int maxNumWidthTiles = 15;
    const int maxNumHeightTiles = 15;
    const int maxNumBatchTiles = _co._outputDims.get(Dim::N, 1);

    const auto outputTileInitial = dirTiling.getOutputTileDims();
    const auto inputTileInitial = dirTiling.getInputTileDims();

    const auto minInputTileDimW = std::max(8, _co._kernelSizeX);
    const auto minInputTileDimH = _co._kernelSizeY;

    //  const DimValues &splitOver = dirTiling.splitOverTensorDims();
    const auto direction = dirTiling.getDirection();

    for (int numBatchTiles = 1; numBatchTiles <= maxNumBatchTiles; numBatchTiles++) {
        //
        // Filter-out misaligned SoN tiles.
        //

        if (outputTileInitial[Dim::N] % numBatchTiles != 0) {
            continue;
        }

        auto tileSizeDimN = outputTileInitial[Dim::N] / numBatchTiles;

        for (int numWidthTiles = 1; numWidthTiles <= maxNumWidthTiles; numWidthTiles++) {
            // const int tileSizeDimW = divUp(splitOver[Dim::W], numWidthTiles);
            int tileSizeDimW = divUp(_co._inputDims[Dim::W], numWidthTiles);

            //
            // Filter-out too small SoW tiles.
            //

            if (numWidthTiles > 1 && direction == Direction::INPUT_TO_OUTPUT) {
                tileSizeDimW = divUp(tileSizeDimW, _co._kernelStride) * _co._kernelStride;

                if (tileSizeDimW < minInputTileDimW) {
                    break;
                }
            }

            for (int numHeightTiles = 1; numHeightTiles <= maxNumHeightTiles ; numHeightTiles++) {
                // const int tileSizeDimH = divUp(splitOver[Dim::H], numHeightTiles);
                int tileSizeDimH = divUp(_co._inputDims[Dim::H], numHeightTiles);

                if (direction == Direction::INPUT_TO_OUTPUT) {
                    tileSizeDimH = divUp(tileSizeDimH, _co._kernelStride) * _co._kernelStride;
                }

                //
                // Filter-out too small SoH tiles.
                //

                if (numHeightTiles > 1 && direction == Direction::INPUT_TO_OUTPUT) {
                    tileSizeDimH = divUp(tileSizeDimH, _co._kernelStride) * _co._kernelStride;

                    if (tileSizeDimH < minInputTileDimH) {
                        break;
                    }
                }

                //
                // Try current tile size.
                //

                dirTiling.resetInputTileDims(inputTileInitial);
                dirTiling.resetOutputTileDims(outputTileInitial);

                dirTiling.setInputNOutputTileDimensions(tileSizeDimW, tileSizeDimH, tileSizeDimN);


                //
                // Check that tiling is valid.
                //

                const auto heightTiles = calcHeightTilesP(_co, dirTiling.getOutputTileDims(),
                                                         dirTiling.useCeil());
                const auto widthTiles = calcWidthTilesP(_co, dirTiling.getOutputTileDims(), dirTiling.useCeil());

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
                        fullOutputTileDims.set(Dim::C, dirTiling.getOutputTileDims()[Dim::C]);
                        fullOutputTileDims.set(Dim::N, dirTiling.getOutputTileDims()[Dim::N]);

                        // TODO: support HCW
                        if (calculateHwBufferSize(fullOutputTileDims) > env.resources.cmxLimit) {
                            isOK = false;
                            break;
                        }

                        //
                        // `linesPerChan` restrictions.
                        //

                        if (heightTile.inputWithJunk < _co._kernelSizeY) {
                            isOK = false;
                            break;
                        }

                        if (!checkPoolingHWRestrictions(
                                widthTile.inputWithJunk,
                                heightTile.inputWithJunk,
                                dirTiling.getInputTileDims()[Dim::C],
                                dirTiling.getOutputTileDims()[Dim::C],
                                _co._kernelSizeX, _co._kernelSizeY, _co._kernelStride)) {
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
                                _co._kernelSizeX, _co._kernelSizeY,
                                _co._kernelStride, _co._kernelStride,
                                _co._paddingLeft, _co._paddingTop);

                        if (pad.enable && (pad.left > 0 || pad.right > 0 || pad.bottom > 0)) {
                            int memPerPlane = alignVal(
                                    fullInputTileDims[Dim::W], 8) * sizeof(fp16_t)
                                              * ((fullInputTileDims[Dim::H] - 1) + (_co._kernelSizeY - 1));
                            int memLimit = pad.bottom > 0 ? 0x800 : 0x1000;
                            if (memPerPlane > memLimit) {
                                isOK = false;
                                break;
                            }
                        }

                        //
                        // Calc tile cost.
                        //
                        const auto& _inputTileDims = dirTiling.getInputTileDims();
                        const auto& _outputTileDims = dirTiling.getOutputTileDims();
                        auto chansPerBlock = 1 << static_cast<int>(HwOpMode::MODE_16_16);
                        solutionCost += 1.0
                                        * ((_inputTileDims[Dim::C] * _inputTileDims[Dim::N]) / chansPerBlock) * _co._kernelSizeX * _co._kernelSizeY
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
                // Put to the pool of best options.
                //

                const int totalNumTiles = numWidthTiles * numHeightTiles * numBatchTiles;

                const TilingOption to =
                        {numWidthTiles, numHeightTiles, numBatchTiles, totalNumTiles, solutionCost};
                tilingOptions.push(to);
            }
        }
    }

    const auto sorted = tilingOptions.sorted();

    if (sorted.size() != 0) {
        const TilingOption& best = sorted.front();
        int inputTileDimW = divUp(_co._inputDims[Dim::W], best.numWidthTiles);
        int inputTileDimH = divUp(_co._inputDims[Dim::H], best.numHeightTiles);
        auto tileDimN = outputTileInitial[Dim::N] / best.numChannelTiles;

        inputTileDimW = divUp(inputTileDimW, _co._kernelStride) * _co._kernelStride;
        inputTileDimH = divUp(inputTileDimH, _co._kernelStride) * _co._kernelStride;

        auto& _inputTileDims = dirTiling.getInputTileDims();
        auto& _outputTileDims = dirTiling.getOutputTileDims();

        _inputTileDims.set(Dim::W, inputTileDimW);
        _inputTileDims.set(Dim::H, inputTileDimH);
        _inputTileDims.set(Dim::N, tileDimN);

        dirTiling.resetOutputTileDims(outputTileInitial);
        _outputTileDims.set(Dim::N, tileDimN);

        dirTiling.correctPlaneSize();
    }

    return sorted;
}

const vpu::HWTilingNS::HWPoolingTileLayoutCut HWPoolingTilingSearcher::tileLayoutCut(const TilingOption &option) const {
    return HWPoolingTileLayoutCut(*_dirTiling, option);
}

SmallVector<HwPlaneTileInfo> calcHeightTilesP(const ConvolutionOptions &_co,
                                              const DimValues &outputTileDims, bool useCeil) {
    SmallVector<HwPlaneTileInfo> heightTiles;

    if (outputTileDims[Dim::H] == _co._outputDims[Dim::H]) {
        HwPlaneTileInfo info;
        info.inputWithJunk = _co._inputDims[Dim::H];
        info.outputWithJunk = _co._outputDims[Dim::H];
        info.outputJunkBefore = 0;
        info.outputJunkAfter = 0;
        info.inputStartIndex = 0;
        info.inputEndIndex = _co._inputDims[Dim::H];
        info.outputStartIndex = 0;
        info.outputEndIndex = _co._outputDims[Dim::H];

        heightTiles.emplace_back(info);
    } else {
        heightTiles = splitIntoPlaneTiles(
                _co._inputDims[Dim::H],
                _co._outputDims[Dim::H],
                _co._kernelSizeY,
                _co._kernelStride,
                _co._paddingTop, _co._paddingBottom,
                outputTileDims[Dim::H],
                useCeil);
    }

    return heightTiles;
}

SmallVector<HwPlaneTileInfo> calcWidthTilesP(const ConvolutionOptions &_co,
                                             const DimValues &outputTileDims, bool useCeil) {
    SmallVector<HwPlaneTileInfo> widthTiles;

    if (outputTileDims[Dim::W] == _co._outputDims[Dim::W]) {
        HwPlaneTileInfo info;
        info.inputWithJunk = _co._inputDims[Dim::W];
        info.outputWithJunk = _co._outputDims[Dim::W];
        info.outputJunkBefore = 0;
        info.outputJunkAfter = 0;
        info.inputStartIndex = 0;
        info.inputEndIndex = _co._inputDims[Dim::W];
        info.outputStartIndex = 0;
        info.outputEndIndex = _co._outputDims[Dim::W];

        widthTiles.emplace_back(info);
    } else {
        widthTiles = splitIntoPlaneTiles(
                _co._inputDims[Dim::W],
                _co._outputDims[Dim::W],
                _co._kernelSizeX,
                _co._kernelStride,
                _co._paddingLeft, _co._paddingRight,
                outputTileDims[Dim::W],
                useCeil);
    }

    return widthTiles;
}

HwPoolTileInfo splitPooling(int outZ) {
    HwPoolTileInfo tiles;
    tiles.mode = HwOpMode::MODE_16_16;
    tiles.numDescr = (outZ + CHANS_PER_DESCR - 1) / CHANS_PER_DESCR;
    tiles.chansPerDescr = CHANS_PER_DESCR;
    return tiles;
}

}  // namespace HWTilingNS

}  // namespace vpu

