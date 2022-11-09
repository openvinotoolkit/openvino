// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <utility>
#include <vpu/middleend/hw/pooling_tiling/hw_pooling_tiler.hpp>

namespace vpu {

namespace HWTilingNS {

class PoolingInputToOutputDirection;
class PoolingOutputToInputDirection;

// Input -> Output case
class PoolingInputToOutputDirection: public GraphDataTiling {
public:
    explicit PoolingInputToOutputDirection(const ConvolutionOptions& convolutionOptions) :
        GraphDataTiling(convolutionOptions, Direction::INPUT_TO_OUTPUT) {}
    PoolingInputToOutputDirection(const PoolingInputToOutputDirection&) = default;

    void initTileSizes() override {
        _useCeil = ceilNeeded();

        _inputTileDims.set(Dim::W, _convolutionOptions._inputDims[Dim::W]);
        _inputTileDims.set(Dim::H, _convolutionOptions._inputDims[Dim::H]);
        _inputTileDims.set(Dim::C, _convolutionOptions._inputDims[Dim::C]);
        _inputTileDims.set(Dim::N, _convolutionOptions._inputDims.get(Dim::N, 1));

        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::C, _convolutionOptions._outputDims[Dim::C]);
        _outputTileDims.set(Dim::N, _convolutionOptions._outputDims.get(Dim::N, 1));

        correctOutputPlaneSize();
    }

    // Input -> Output case
    void setInputNOutputTileDimensions(int tileDimW, int tileDimH, int tileDimN) override {
        _inputTileDims.set(Dim::W, tileDimW);
        _inputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::N, tileDimN);

        _outputTileDims.set(Dim::N, tileDimN);

        correctOutputPlaneSize();
    }

    // Input -> Output case
    void applyTilingOption(const TilingOption& tilingOption) override {
        int tileDimW = divUp(_convolutionOptions._inputDims[Dim::W], tilingOption.numWidthTiles);
        int tileDimH = divUp(_convolutionOptions._inputDims[Dim::H], tilingOption.numHeightTiles);
        const int tileDimN = divUp(_convolutionOptions._inputDims[Dim::N], tilingOption.numChannelTiles);

        if (tilingOption.numWidthTiles > 1) {
            tileDimW = divUp(tileDimW, _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;
        }
        if (tilingOption.numHeightTiles > 1) {
            tileDimH = divUp(tileDimH, _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;
        }

        _inputTileDims.set(Dim::W, tileDimW);
        _inputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::N, tileDimN);

        _outputTileDims.set(Dim::N, tileDimN);

        correctOutputPlaneSize();
    }

    void correctPlaneSize() override {
        correctOutputPlaneSize();
    }

    void correctPlaneSizeAfterPatternMatching() override {
        // noop
    }

    void correctOutputPlaneSize() {
        const auto maxOutputWidth = calcOutputSize(
            _inputTileDims[Dim::W],
            _convolutionOptions._kernelSizeX,
            _convolutionOptions._kernelStride,
            _convolutionOptions._paddingLeft,
            _convolutionOptions._paddingRight,
            _useCeil);

        _outputTileDims.set(Dim::W, std::min(_outputTileDims[Dim::W], maxOutputWidth));

        const auto maxOutputHeight = calcOutputSize(
            _inputTileDims[Dim::H],
            _convolutionOptions._kernelSizeY,
            _convolutionOptions._kernelStride,
            _convolutionOptions._paddingTop,
            _convolutionOptions._paddingBottom,
            _useCeil);

        _outputTileDims.set(Dim::H, std::min(_outputTileDims[Dim::H], maxOutputHeight));
    }

    const DimValues& splitOverTensorDims() override {
        return _convolutionOptions._inputDims;
    }

    bool patternMatching() override {
        return false;
    };

private:
    bool ceilNeeded() {
        const auto getDimension = [](int dimension, int padding_0, int padding_1, int kernel) {
            return dimension + padding_0 + padding_1 - kernel;
        };

        const auto tempX = getDimension(
            _convolutionOptions._inputDims[Dim::W],
            _convolutionOptions._paddingLeft,
            _convolutionOptions._paddingRight,
            _convolutionOptions._kernelSizeX);

        const auto tempY = getDimension(
            _convolutionOptions._inputDims[Dim::H],
            _convolutionOptions._paddingTop,
            _convolutionOptions._paddingBottom,
            _convolutionOptions._kernelSizeY);

        const auto outWidthWithOutCeil  =
            (tempX + _convolutionOptions._kernelStride) / _convolutionOptions._kernelStride;
        const auto outHeightWithOutCeil =
            (tempY + _convolutionOptions._kernelStride) / _convolutionOptions._kernelStride;

        const auto outWidthWithCeil =  static_cast<int>(
            std::ceil(static_cast<double>(tempX) / _convolutionOptions._kernelStride + 1));
        const auto outHeightWithCeil = static_cast<int>(
            std::ceil(static_cast<double>(tempY) / _convolutionOptions._kernelStride + 1));

        if ((_convolutionOptions._outputDims[Dim::W] != outWidthWithCeil) &&
            (_convolutionOptions._outputDims[Dim::W] != outWidthWithOutCeil)) {
            VPU_THROW_EXCEPTION << "Internal error: Output in " << _convolutionOptions._stageName
                                << " has incorrect width dimension. Expected: " << outWidthWithCeil
                                << " or " << outWidthWithOutCeil
                                << " Actual: " << _convolutionOptions._outputDims[Dim::W];
        }

        if ((_convolutionOptions._outputDims[Dim::H] != outHeightWithCeil) &&
            (_convolutionOptions._outputDims[Dim::H] != outHeightWithOutCeil)) {
            VPU_THROW_EXCEPTION << "Internal error: Output in " << _convolutionOptions._stageName
                                << " has incorrect height dimension. Expected: " << outHeightWithCeil
                                << " or " << outHeightWithOutCeil
                                << " Actual: " << _convolutionOptions._outputDims[Dim::H];
        }

        return ((_convolutionOptions._origOutputDims[Dim::W] == outWidthWithCeil) &&
                (_convolutionOptions._origOutputDims[Dim::H] == outHeightWithCeil));
    }
};

// Output -> Input case
class PoolingOutputToInputDirection: public GraphDataTiling {
public:
    explicit PoolingOutputToInputDirection(const ConvolutionOptions& convolutionOptions) :
        GraphDataTiling(convolutionOptions, Direction::INPUT_TO_OUTPUT) {}
    PoolingOutputToInputDirection(const PoolingOutputToInputDirection&) = default;

    void initTileSizes() override {
        _useCeil = false;   // no ceiling needed for PoolingOutputToInputDirection

        _inputTileDims.set(Dim::W, _convolutionOptions._inputDims[Dim::W]);
        _inputTileDims.set(Dim::H, _convolutionOptions._inputDims[Dim::H]);
        _inputTileDims.set(Dim::C, _convolutionOptions._inputDims[Dim::C]);
        _inputTileDims.set(Dim::N, _convolutionOptions._inputDims.get(Dim::N, 1));

        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::C, _convolutionOptions._outputDims[Dim::C]);
        _outputTileDims.set(Dim::N, _convolutionOptions._outputDims.get(Dim::N, 1));

        correctInputPlaneSize();
    }

    // Output -> Input case
    void setInputNOutputTileDimensions(int tileDimW, int tileDimH, int tileDimN) override {
        _outputTileDims.set(Dim::W, tileDimW);
        _outputTileDims.set(Dim::H, tileDimH);
        _outputTileDims.set(Dim::N, tileDimN);

        _inputTileDims.set(Dim::N, tileDimN);

        correctInputPlaneSize();
    }

    // Output -> Input case
    void applyTilingOption(const TilingOption& tilingOption) override {
        const int tileDimW = divUp(_convolutionOptions._outputDims[Dim::W], tilingOption.numWidthTiles);
        const int tileDimH = divUp(_convolutionOptions._outputDims[Dim::H], tilingOption.numHeightTiles);
        const int tileDimN = divUp(_convolutionOptions._inputDims[Dim::N], tilingOption.numChannelTiles);

        _outputTileDims.set(Dim::W, tileDimW);
        _outputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::N, tileDimN);

        correctInputPlaneSize();
    }

    void correctPlaneSize() override {
        correctInputPlaneSize();
    }

    void correctPlaneSizeAfterPatternMatching() override {
        // noop
    }

    void correctInputPlaneSize() {
        const auto maxInputWidth = calcInputSize(
            _outputTileDims[Dim::W],
            _convolutionOptions._kernelSizeX,
            _convolutionOptions._kernelStride,
            _convolutionOptions._paddingLeft,
            _convolutionOptions._paddingRight);

        _inputTileDims.set(Dim::W, std::min(_inputTileDims[Dim::W], maxInputWidth));

        const auto maxInputHeight = calcInputSize(
            _outputTileDims[Dim::H],
            _convolutionOptions._kernelSizeY,
            _convolutionOptions._kernelStride,
            _convolutionOptions._paddingTop,
            _convolutionOptions._paddingBottom);

        _inputTileDims.set(Dim::H, std::min(_inputTileDims[Dim::H], maxInputHeight));
    }

    const DimValues& splitOverTensorDims() override {
        return _convolutionOptions._outputDims;
    }

    bool patternMatching() override {
        return false;
    };

private:
    PoolingOutputToInputDirection& operator=(const PoolingOutputToInputDirection& src) { return *this; }
};


HWPoolingTiler::HWPoolingTiler(const ConvolutionOptions& convolutionOptions, const Direction& direction,
                               std::size_t maxTilingOptions) :
    _convolutionOptions(std::move(convolutionOptions)),
    _searcher(_convolutionOptions, direction, maxTilingOptions) {
    _tilingPossible = tileForHW();
}

bool HWPoolingTiler::tileForHW() {
    const std::vector<TilingOption>& tilingOptions = _searcher.tilingOptions();
    if (tilingOptions.empty()) {
        return false;
    }

    for (const TilingOption& tilingOption : tilingOptions) {
        const HWPoolingTileLayoutCut tileLayoutCut = _searcher.tileLayoutCut(tilingOption);
        if (tileLayoutCut.tileCutPossible()) {
            _hwTilings.push_back(tileLayoutCut.hwTiling());
        }
    }

    return _hwTilings.size() != 0;
}

std::unique_ptr<GraphDataTiling> PoolGraphDataTilingFactory::makeDirTiling(const ConvolutionOptions& convolutionOptions,
                                                                           const Direction& direction) {
    if (direction == Direction::INPUT_TO_OUTPUT) {
        return std::unique_ptr<GraphDataTiling>(new PoolingInputToOutputDirection(convolutionOptions));
    } else if (direction == Direction::OUTPUT_TO_INPUT) {
        return std::unique_ptr<GraphDataTiling>(new PoolingOutputToInputDirection(convolutionOptions));
    }
    IE_THROW() << "Unsupported direction";
}

std::unique_ptr<GraphDataTiling> PoolGraphDataTilingFactory::makeDirTiling(const GraphDataTiling& graphDataTiling) {
    if (graphDataTiling.getDirection() == Direction::INPUT_TO_OUTPUT) {
        return std::unique_ptr<GraphDataTiling>(
                new PoolingInputToOutputDirection(dynamic_cast<const PoolingInputToOutputDirection&>(graphDataTiling)));
    } else if (graphDataTiling.getDirection() == Direction::OUTPUT_TO_INPUT) {
        return std::unique_ptr<GraphDataTiling>(
                new PoolingOutputToInputDirection(dynamic_cast<const PoolingOutputToInputDirection&>(graphDataTiling)));
    }
    IE_THROW() << "Unsupported direction";
}

//
// Looks for the optimal tiling accordingly to the cost function. Modifies dimensions in dirTiling during search.
//
std::vector<TilingOption> HWPoolingTilingSearcher::selectBetterTiling() const {
    const auto& env = CompileEnv::get();

    auto& dirTiling = *_dirTiling;
    FixedMaxHeap<TilingOption> tilingOptions(_maxTilingOptions);

    // TODO: estimate this numbers
    const int maxNumWidthTiles = 15;
    const int maxNumHeightTiles = 15;
    const int maxNumBatchTiles = _convolutionOptions._outputDims.get(Dim::N, 1);

    const auto outputTileInitial = dirTiling.getOutputTileDims();
    const auto inputTileInitial = dirTiling.getInputTileDims();

    const auto minInputTileDimW = std::max(8, _convolutionOptions._kernelSizeX);
    const auto minInputTileDimH = _convolutionOptions._kernelSizeY;

    const auto& splitOver = dirTiling.splitOverTensorDims();
    const auto direction = dirTiling.getDirection();
    const auto cmxLimit = env.resources.tilingCMXLimit;

    for (int numBatchTiles = 1; numBatchTiles <= maxNumBatchTiles; numBatchTiles++) {
        //
        // Filter-out misaligned SoN tiles.
        //

        if (outputTileInitial[Dim::N] % numBatchTiles != 0) {
            continue;
        }

        const auto tileSizeDimN = outputTileInitial[Dim::N] / numBatchTiles;

        for (int numWidthTiles = 1; numWidthTiles <= maxNumWidthTiles; numWidthTiles++) {
            int tileSizeDimW = divUp(splitOver[Dim::W], numWidthTiles);

            //
            // Filter-out too small SoW tiles.
            //

            if (numWidthTiles > 1 && direction == Direction::INPUT_TO_OUTPUT) {
                tileSizeDimW =
                    divUp(tileSizeDimW, _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;

                if (tileSizeDimW < minInputTileDimW) {
                    break;
                }
            }

            for (int numHeightTiles = 1; numHeightTiles <= maxNumHeightTiles ; numHeightTiles++) {
                int tileSizeDimH = divUp(splitOver[Dim::H], numHeightTiles);

                //
                // Filter-out too small SoH tiles.
                //

                if (numHeightTiles > 1 && direction == Direction::INPUT_TO_OUTPUT) {
                    tileSizeDimH =
                        divUp(tileSizeDimH, _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;

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

                const auto heightTiles = calcHeightTilesP(
                    _convolutionOptions,
                    dirTiling.getOutputTileDims(),
                    dirTiling.useCeil());

                const auto widthTiles = calcWidthTilesP(
                    _convolutionOptions,
                    dirTiling.getOutputTileDims(),
                    dirTiling.useCeil());

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
                        if (calculateHwBufferSize(fullOutputTileDims) > cmxLimit) {
                            isOK = false;
                            break;
                        }

                        //
                        // `linesPerChan` restrictions.
                        //

                        if (heightTile.inputWithJunk < _convolutionOptions._kernelSizeY) {
                            isOK = false;
                            break;
                        }

                        if (!checkPoolingHWRestrictions(
                                widthTile.inputWithJunk,
                                heightTile.inputWithJunk,
                                dirTiling.getInputTileDims()[Dim::C],
                                dirTiling.getOutputTileDims()[Dim::C],
                                _convolutionOptions._kernelSizeX,
                                _convolutionOptions._kernelSizeY,
                                _convolutionOptions._kernelStride)) {
                            isOK = false;
                            break;
                        }

                        //
                        // Replicate padding in case of large input plane - #-16783.
                        //

                        DimValues fullInputTileDims;
                        fullInputTileDims.set(Dim::W, widthTile.inputWithJunk);
                        fullInputTileDims.set(Dim::H, heightTile.inputWithJunk);

                        const auto pad = getHwPaddingInfo(
                            fullInputTileDims,
                            fullOutputTileDims,
                            _convolutionOptions._kernelSizeX,
                            _convolutionOptions._kernelSizeY,
                            _convolutionOptions._kernelStride,
                            _convolutionOptions._kernelStride,
                            _convolutionOptions._paddingLeft,
                            _convolutionOptions._paddingTop);

                        if (pad.enable && (pad.left > 0 || pad.right > 0 || pad.bottom > 0)) {
                            const auto memPerPlane =
                                alignVal(fullInputTileDims[Dim::W], 8) * sizeof(fp16_t)
                                * ((fullInputTileDims[Dim::H] - 1) + (_convolutionOptions._kernelSizeY - 1));

                            int memLimit = pad.bottom > 0 ? 0x800 : 0x1000;
                            if (memPerPlane > memLimit) {
                                isOK = false;
                                break;
                            }
                        }

                        //
                        // Calc tile cost
                        //

                        const auto& _inputTileDims = dirTiling.getInputTileDims();
                        const auto& _outputTileDims = dirTiling.getOutputTileDims();
                        const auto chansPerBlock = 1ULL << static_cast<std::size_t>(HwOpMode::MODE_16_16);
                        solutionCost += 1.0
                                        * ((_inputTileDims[Dim::C] * _inputTileDims[Dim::N]) / chansPerBlock)
                                        * _convolutionOptions._kernelSizeX
                                        * _convolutionOptions._kernelSizeY
                                        * numBatchTiles;

                        // Alignment for output
                        if ((widthTile.outputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                            solutionCost += static_cast<double>(widthTile.outputWithJunk)
                                            * heightTile.outputWithJunk
                                            * _outputTileDims[Dim::C]
                                            * _outputTileDims[Dim::N];
                        }

                        // Alignment for input
                        if ((widthTile.inputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                            solutionCost += static_cast<double>(widthTile.inputWithJunk)
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

                tilingOptions.push({numWidthTiles, numHeightTiles, numBatchTiles, totalNumTiles, solutionCost});
            }
        }
    }

    const auto sorted = tilingOptions.sorted();
    if (!sorted.empty()) {
        const TilingOption& best = sorted.front();
        int inputTileDimW = divUp(_convolutionOptions._inputDims[Dim::W], best.numWidthTiles);
        int inputTileDimH = divUp(_convolutionOptions._inputDims[Dim::H], best.numHeightTiles);
        auto tileDimN = outputTileInitial[Dim::N] / best.numChannelTiles;

        if (best.numWidthTiles > 1) {
            inputTileDimW = divUp(inputTileDimW, _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;
        }
        if (best.numHeightTiles > 1) {
            inputTileDimH = divUp(inputTileDimH, _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;
        }

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

vpu::HWTilingNS::HWPoolingTileLayoutCut HWPoolingTilingSearcher::tileLayoutCut(const TilingOption& option) const {
    return HWPoolingTileLayoutCut(*_dirTiling, option);
}

SmallVector<HwPlaneTileInfo> calcHeightTilesP(const ConvolutionOptions& convolutionOptions,
                                              const DimValues& outputTileDims, bool useCeil) {
    SmallVector<HwPlaneTileInfo> heightTiles;

    if (outputTileDims[Dim::H] == convolutionOptions._outputDims[Dim::H]) {
        HwPlaneTileInfo info;
        info.inputWithJunk = convolutionOptions._inputDims[Dim::H];
        info.outputWithJunk = convolutionOptions._outputDims[Dim::H];
        info.outputJunkBefore = 0;
        info.outputJunkAfter = 0;
        info.inputStartIndex = 0;
        info.inputEndIndex = convolutionOptions._inputDims[Dim::H];
        info.outputStartIndex = 0;
        info.outputEndIndex = convolutionOptions._outputDims[Dim::H];

        heightTiles.emplace_back(info);
    } else {
        heightTiles = splitIntoPlaneTiles(
            convolutionOptions._inputDims[Dim::H],
            convolutionOptions._outputDims[Dim::H],
            convolutionOptions._kernelSizeY,
            convolutionOptions._kernelStride,
            convolutionOptions._paddingTop, convolutionOptions._paddingBottom,
            outputTileDims[Dim::H],
            useCeil);
    }

    return heightTiles;
}

SmallVector<HwPlaneTileInfo> calcWidthTilesP(const ConvolutionOptions& convolutionOptions,
                                             const DimValues& outputTileDims, bool useCeil) {
    SmallVector<HwPlaneTileInfo> widthTiles;

    if (outputTileDims[Dim::W] == convolutionOptions._outputDims[Dim::W]) {
        HwPlaneTileInfo info;
        info.inputWithJunk = convolutionOptions._inputDims[Dim::W];
        info.outputWithJunk = convolutionOptions._outputDims[Dim::W];
        info.outputJunkBefore = 0;
        info.outputJunkAfter = 0;
        info.inputStartIndex = 0;
        info.inputEndIndex = convolutionOptions._inputDims[Dim::W];
        info.outputStartIndex = 0;
        info.outputEndIndex = convolutionOptions._outputDims[Dim::W];

        widthTiles.emplace_back(info);
    } else {
        widthTiles = splitIntoPlaneTiles(
            convolutionOptions._inputDims[Dim::W],
            convolutionOptions._outputDims[Dim::W],
            convolutionOptions._kernelSizeX,
            convolutionOptions._kernelStride,
            convolutionOptions._paddingLeft, convolutionOptions._paddingRight,
            outputTileDims[Dim::W],
            useCeil);
    }

    return widthTiles;
}

HwPoolTileInfo splitPooling(int outZ) {
    HwPoolTileInfo tiles;
    tiles.mode = HwOpMode::MODE_16_16;
    tiles.numDescr = (outZ + CHANNELS_PER_DESCRIPTOR - 1) / CHANNELS_PER_DESCRIPTOR;
    tiles.chansPerDescr = CHANNELS_PER_DESCRIPTOR;
    return tiles;
}

}  // namespace HWTilingNS

}  // namespace vpu

