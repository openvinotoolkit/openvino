// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <utility>
#include <vpu/passes/hw_conv_tiling/hw_convolution_tiler.hpp>

namespace vpu {

namespace HWTilingNS {

bool operator<(const TilingOption& a, const TilingOption& b) {
    return a.cost < b.cost || (isDoubleEqual(a.cost, b.cost) && a.totalNumTiles < b.totalNumTiles);
}

void correctOutputPlaneSizeF(const ConvolutionOptions &_co,  bool _useCeil,
        const DimValues &_inputTileDims, DimValues &_outputTileDims) {
    int maxOutputWidth = calcOutputSize(_inputTileDims[Dim::W], _co._kernelSizeX, _co._kernelStride,
                                        _co._paddingLeft, _co._paddingRight, _useCeil);
    if (_co._withPool) {
        maxOutputWidth /= 2;
    }
    _outputTileDims.set(Dim::W, std::min(_outputTileDims[Dim::W], maxOutputWidth));

    int maxOutputHeight = calcOutputSize(_inputTileDims[Dim::H], _co._kernelSizeY, _co._kernelStride,
                                         _co._paddingTop, _co._paddingBottom, _useCeil);
    if (_co._withPool) {
        maxOutputHeight /= 2;
    }
    _outputTileDims.set(Dim::H, std::min(_outputTileDims[Dim::H], maxOutputHeight));
}

class ConvInputToOutputDirection;
class ConvOutputToInputDirection;

// Input -> Output case
class ConvInputToOutputDirection: public GraphDataTiling {
public:
    explicit ConvInputToOutputDirection(const ConvolutionOptions &co): GraphDataTiling(co, Direction::INPUT_TO_OUTPUT) {}
    ConvInputToOutputDirection(const ConvInputToOutputDirection &other): GraphDataTiling(other) {}
    void initTileSizes() override {
        _useCeil = ceilNeeded();

        _inputTileDims.set(Dim::W, std::min(CNN_MAX_INPUT_WIDTH, _co._inputDims[Dim::W]));
        _inputTileDims.set(Dim::H, std::min(CNN_MAX_INPUT_HEIGHT, _co._inputDims[Dim::H]));
        _inputTileDims.set(Dim::C, std::min(CNN_MAX_INPUT_CHANNELS, _co._inputDims[Dim::C]));

        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::C, _co._outputDims[Dim::C]);

        correctOutputPlaneSize();
    }

    // Input -> Output case
    void setInputNOutputTileDimensions(const int tileDimW, const int tileDimH, const int tileDimC) override {
        _inputTileDims.set(Dim::W, tileDimW);
        _inputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::C, tileDimC);

        correctOutputPlaneSize();
    }

    // Input -> Output case
    void applyTilingOption(const TilingOption &tilingOption) override {
        int tileDimW = divUp(_co._inputDims[Dim::W], tilingOption.numWidthTiles);
        int tileDimH = divUp(_co._inputDims[Dim::H], tilingOption.numHeightTiles);
        const int tileDimC = divUp(_co._inputDims[Dim::C], tilingOption.numChannelTiles);

        tileDimW = divUp(tileDimW, _co._kernelStride) * _co._kernelStride;
        tileDimH = divUp(tileDimH, _co._kernelStride) * _co._kernelStride;

        _inputTileDims.set(Dim::W, tileDimW);
        _inputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::C, tileDimC);

        correctOutputPlaneSize();
    }

    void correctPlaneSize() override {
        correctOutputPlaneSize();
    }

    void correctPlaneSizeAfterPatternMatching() override {
        correctOutputPlaneSize();
    }

    void correctOutputPlaneSize() {
        correctOutputPlaneSizeF(_co, _useCeil, _inputTileDims, _outputTileDims);
    }

    const DimValues &splitOverTensorDims() override {
        return _co._inputDims;
    }

    bool patternMatching() override {
        return GraphDataTiling::patternMatching();
    }

private:
};

// Output -> Input case
class ConvOutputToInputDirection: public GraphDataTiling {
public:
    explicit ConvOutputToInputDirection(const ConvolutionOptions &co): GraphDataTiling(co, Direction::OUTPUT_TO_INPUT) {}
    ConvOutputToInputDirection(const ConvOutputToInputDirection &other): GraphDataTiling(other) {}
    void initTileSizes() override {
        _useCeil = false;   // no ceiling needed for calculating input sizes from output sizes

        _outputTileDims.set(Dim::W, std::min(CNN_MAX_INPUT_WIDTH, _co._outputDims[Dim::W]));
        _outputTileDims.set(Dim::H, std::min(CNN_MAX_INPUT_HEIGHT, _co._outputDims[Dim::H]));
        _outputTileDims.set(Dim::C, _co._outputDims[Dim::C]);

        _inputTileDims.set(Dim::W, std::min(CNN_MAX_INPUT_WIDTH, _co._inputDims[Dim::W]));
        _inputTileDims.set(Dim::H, std::min(CNN_MAX_INPUT_HEIGHT, _co._inputDims[Dim::H]));
        _inputTileDims.set(Dim::C, std::min(CNN_MAX_INPUT_CHANNELS, _co._inputDims[Dim::C]));

        correctInputPlaneSize();
    }
    // Output -> Input case
    void setInputNOutputTileDimensions(const int tileDimW, const int tileDimH, const int tileDimC) override {
        _outputTileDims.set(Dim::W, tileDimW);
        _outputTileDims.set(Dim::H, tileDimH);
        _outputTileDims.set(Dim::C, tileDimC);

        correctInputPlaneSize();
    }

    // Output -> Input case
    void applyTilingOption(const TilingOption &tilingOption) override {
        const int tileDimW = divUp(_co._outputDims[Dim::W], tilingOption.numWidthTiles);
        const int tileDimH = divUp(_co._outputDims[Dim::H], tilingOption.numHeightTiles);
        // split only input tensor over C dim
        const int tileDimC = divUp(_co._inputDims[Dim::C], tilingOption.numChannelTiles);

        _outputTileDims.set(Dim::W, tileDimW);
        _outputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::C, tileDimC);

        correctInputPlaneSize();
    }

    void correctPlaneSize() override {
        correctInputPlaneSize();
    }

    void correctPlaneSizeAfterPatternMatching() override {
        correctOutputPlaneSizeF(_co, _useCeil, _inputTileDims, _outputTileDims);
    }

    void correctInputPlaneSize() {
        int maxInputWidth = calcInputSize(_outputTileDims[Dim::W], _co._kernelSizeX, _co._kernelStride, _co._paddingLeft,
                                          _co._paddingRight);
        if (_co._withPool) {
            maxInputWidth *= 2;
        }
        _inputTileDims.set(Dim::W, std::min(_inputTileDims[Dim::W], maxInputWidth));

        int maxInputHeight = calcInputSize(_outputTileDims[Dim::H], _co._kernelSizeY, _co._kernelStride, _co._paddingTop,
                                           _co._paddingBottom);
        if (_co._withPool) {
            maxInputHeight *= 2;
        }
        _inputTileDims.set(Dim::H, std::min(_inputTileDims[Dim::H], maxInputHeight));
    }

    const DimValues &splitOverTensorDims() override {
        return _co._outputDims;
    }

    bool patternMatching() override {
        return false;
    }
};

HWConvolutionTiler::HWConvolutionTiler(const ConvolutionOptions &co,
                   Direction direction,
                   size_t maxTilingOptions) :
        _co(co),
        _searcher(_co, direction, maxTilingOptions) {
    _tilingPossible = tileForHW();
}

bool HWConvolutionTiler::tileForHW() {
    const std::vector<TilingOption> &tilingOptions = _searcher.tilingOptions();
    if (tilingOptions.empty()) {
            return false;
    }

    for (const TilingOption &tilingOption : tilingOptions) {
        const HWConvolutionTileLayoutCut tileLayoutCut = _searcher.tileLayoutCut(tilingOption);
        if (tileLayoutCut.tileCutPossible()) {
            _hwTilings.push_back(tileLayoutCut.hwTiling());
        }
    }

    return _hwTilings.size() != 0;
}

bool GraphDataTiling::ceilNeeded() const {
    int tempX = _co._inputDims[Dim::W] + _co._paddingLeft + _co._paddingRight - _co._kernelSizeX;
    int tempY = _co._inputDims[Dim::H] + _co._paddingTop + _co._paddingBottom - _co._kernelSizeY;

    int outWidthWithOutCeil = (tempX + _co._kernelStride) / _co._kernelStride;
    int outHeightWithOutCeil = (tempY + _co._kernelStride) / _co._kernelStride;

    int outWidthWithCeil = static_cast<int>(std::ceil(static_cast<double>(tempX) / _co._kernelStride + 1));
    int outHeightWithCeil = static_cast<int>(std::ceil(static_cast<double>(tempY) / _co._kernelStride + 1));

    if ((_co._origOutputDims[Dim::W] != outWidthWithCeil) && (_co._origOutputDims[Dim::W] != outWidthWithOutCeil)) {
        VPU_THROW_EXCEPTION
                << "Internal error: Output in " << _co._stageName << " has incorrect width dimension. Expected: "
                << outWidthWithCeil << " or " << outWidthWithOutCeil << " Actual: " << _co._origOutputDims[Dim::W];
    }

    if ((_co._origOutputDims[Dim::H] != outHeightWithCeil) && (_co._origOutputDims[Dim::H] != outHeightWithOutCeil)) {
        VPU_THROW_EXCEPTION
                << "Internal error: Output in " << _co._stageName << " has incorrect height dimension. Expected: "
                << outHeightWithCeil << " or " << outHeightWithOutCeil << " Actual: " << _co._origOutputDims[Dim::H];
    }

    if ((_co._origOutputDims[Dim::W] == outWidthWithOutCeil) && (_co._origOutputDims[Dim::H] == outHeightWithOutCeil)) {
        return false;
    } else {
        return true;
    }
}

bool GraphDataTiling::patternMatching() {
    if (!_co._withPool &&
        _co._kernelSizeX == 3 && _co._kernelSizeY == 3 && _co._paddingLeft == 1 && _co._paddingRight == 1  &&
        _co._paddingTop == 1 && _co._paddingBottom == 1  && _co._kernelStride == 1 &&
        _co._inputDims[Dim::C] == 512 && _co._inputDims[Dim::H] == 28 && _co._inputDims[Dim::W] == 28 &&
        _co._outputDims[Dim::C] == 512) {
        _inputTileDims.set(Dim::H, 28);
        _inputTileDims.set(Dim::C, 172);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (!_co._withPool &&
        _co._kernelSizeX == 3 && _co._kernelSizeY == 3 && _co._paddingLeft == 1 && _co._paddingRight == 1  &&
        _co._paddingTop == 1 && _co._paddingBottom == 1  && _co._kernelStride == 1 &&
        _co._inputDims[Dim::C] == 256 && _co._inputDims[Dim::H] == 56 && _co._inputDims[Dim::W] == 56 &&
        _co._outputDims[Dim::C] == 256) {
        _inputTileDims.set(Dim::H, 30);
        _inputTileDims.set(Dim::C, 128);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (!_co._withPool &&
        _co._kernelSizeX == 3 && _co._kernelSizeY == 3 && _co._paddingLeft == 1 && _co._paddingRight == 1  &&
        _co._paddingTop == 1 && _co._paddingBottom == 1  && _co._kernelStride == 1 &&
        _co._inputDims[Dim::C] == 64 && _co._inputDims[Dim::H] == 224 && _co._inputDims[Dim::W] == 224 &&
        _co._outputDims[Dim::C] == 64) {
        _inputTileDims.set(Dim::H, 82);
        _inputTileDims.set(Dim::W, 82);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (_co._inputDims[Dim::C] == 512 &&
        _co._inputDims[Dim::H] == 7 &&
        _co._inputDims[Dim::W] == 7 &&
        _co._outputDims[Dim::C] == 4096) {
        _inputTileDims.set(Dim::C, 64);

        return true;
    }

    if (!_co._withPool &&
        _co._kernelSizeX == 3 && _co._kernelSizeY == 3 && _co._paddingLeft == 1 && _co._paddingRight == 1  &&
        _co._paddingTop == 1 && _co._paddingBottom == 1  && _co._kernelStride == 1 &&
        _co._inputDims[Dim::C] == 128 && _co._inputDims[Dim::H] == 112 && _co._inputDims[Dim::W] == 112 &&
        _co._outputDims[Dim::C] == 128) {
        _inputTileDims.set(Dim::H, 32);
        _inputTileDims.set(Dim::W, 112);
        _inputTileDims.set(Dim::C, 32);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (_co._inputDims[Dim::C] == 1088 &&
        _co._inputDims[Dim::H] == 17 &&
        _co._inputDims[Dim::W] == 17 &&
        (_co._outputDims[Dim::C] == 128 || _co._outputDims[Dim::C] == 192)) {
        _inputTileDims.set(Dim::H, 17);
        _inputTileDims.set(Dim::C, 544);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (_co._inputDims[Dim::C] == 1024 &&
        _co._inputDims[Dim::H] == 17 &&
        _co._inputDims[Dim::W] == 17 &&
        _co._outputDims[Dim::C] == 384) {
        _inputTileDims.set(Dim::H, 17);
        _inputTileDims.set(Dim::C, 512);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (!_co._withPool &&
        _co._kernelSizeX == 3 && _co._kernelSizeY == 3 && _co._paddingLeft == 0 && _co._paddingRight == 0  &&
        _co._paddingTop == 0 && _co._paddingBottom == 0  && _co._kernelStride == 2 &&
        _co._inputDims[Dim::C] == 384 && _co._inputDims[Dim::H] == 35 && _co._inputDims[Dim::W] == 35 &&
        _co._outputDims[Dim::C] == 384) {
        _inputTileDims.set(Dim::C, 194);
        _inputTileDims.set(Dim::H, 35);
        _inputTileDims.set(Dim::W, 35);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (_co._inputDims[Dim::C] == 192 &&
        _co._inputDims[Dim::H] == 71 &&
        _co._inputDims[Dim::W] == 71 &&
        _co._outputDims[Dim::H] == 35) {
        _inputTileDims.set(Dim::W, 71);
        _inputTileDims.set(Dim::C, 96);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (!_co._withPool &&
        _co._inputDims[Dim::C] == 256 &&
        _co._inputDims[Dim::H] == 128 &&
        _co._inputDims[Dim::W] == 128 &&
        _co._outputDims[Dim::C] == 256) {
        _inputTileDims.set(Dim::W, 128);
        _inputTileDims.set(Dim::H, 15);
        _inputTileDims.set(Dim::C, 64);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (!_co._withPool &&
        _co._inputDims[Dim::C] == 512 &&
        _co._inputDims[Dim::H] == 64 &&
        _co._inputDims[Dim::W] == 64 &&
        _co._outputDims[Dim::C] == 512) {
        _inputTileDims.set(Dim::W, 64);
        _inputTileDims.set(Dim::H, 10);
        _inputTileDims.set(Dim::C, 128);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (!_co._withPool &&
        _co._kernelSizeX == 1 && _co._kernelSizeY == 1 && _co._paddingLeft == 0 && _co._paddingRight == 0  &&
        _co._paddingTop == 0 && _co._paddingBottom == 0  && _co._kernelStride == 1 &&
        _co._inputDims[Dim::C] == 384 &&
        _co._inputDims[Dim::H] == 56 &&
        _co._inputDims[Dim::W] == 56 &&
        _co._outputDims[Dim::C] == 64) {
        _inputTileDims.set(Dim::C, 384);
        _inputTileDims.set(Dim::H, 56);
        _inputTileDims.set(Dim::W, 20);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (!_co._withPool &&
        _co._kernelSizeX == 1 && _co._kernelSizeY == 1 && _co._paddingLeft == 0 && _co._paddingRight == 0  &&
        _co._paddingTop == 0 && _co._paddingBottom == 0  && _co._kernelStride == 1 &&
        _co._inputDims[Dim::C] == 2112 &&
        _co._inputDims[Dim::H] == 14 &&
        _co._inputDims[Dim::W] == 14 &&
        _co._outputDims[Dim::C] == 1056) {
        _inputTileDims.set(Dim::C, 556);
        _inputTileDims.set(Dim::H, 14);
        _inputTileDims.set(Dim::W, 14);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (!_co._withPool &&
        _co._kernelSizeX == 3 && _co._kernelSizeY == 3 && _co._paddingLeft == 1 && _co._paddingRight == 1  &&
        _co._paddingTop == 1 && _co._paddingBottom == 1  && _co._kernelStride == 2 &&
        _co._inputDims[Dim::C] == 256 &&
        _co._inputDims[Dim::H] == 52 &&
        _co._inputDims[Dim::W] == 52 &&
        _co._outputDims[Dim::C] == 512) {
        _inputTileDims.set(Dim::C, 128);
        _inputTileDims.set(Dim::H, 52);
        _inputTileDims.set(Dim::W, 52);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    if (!_co._withPool &&
        _co._kernelSizeX == 3 && _co._kernelSizeY == 3 && _co._paddingLeft == 1 && _co._paddingRight == 1  &&
        _co._paddingTop == 1 && _co._paddingBottom == 1  && _co._kernelStride == 1 &&
        _co._inputDims[Dim::C] == 256 &&
        _co._inputDims[Dim::H] == 23 &&
        _co._inputDims[Dim::W] == 23 &&
        _co._outputDims[Dim::C] == 640) {
        _inputTileDims.set(Dim::C, 256);
        _inputTileDims.set(Dim::H, 14);
        _inputTileDims.set(Dim::W, 23);
        _outputTileDims.set(Dim::H, _co._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _co._outputDims[Dim::W]);

        return true;
    }

    return false;
}

std::unique_ptr<GraphDataTiling> ConvGraphDataTilingFactory::makeDirTiling(const ConvolutionOptions &co,
        Direction direction) {
    if (direction == Direction::INPUT_TO_OUTPUT) {
        return std::unique_ptr<GraphDataTiling>(new ConvInputToOutputDirection(co));
    } else if (direction == Direction::OUTPUT_TO_INPUT) {
        return std::unique_ptr<GraphDataTiling>(new ConvOutputToInputDirection(co));
    } else {
        IE_ASSERT(false) << "Unsupported direction";
    }
}

std::unique_ptr<GraphDataTiling> ConvGraphDataTilingFactory::makeDirTiling(const GraphDataTiling &o) {
    if (o.getDirection() == Direction::INPUT_TO_OUTPUT) {
        return std::unique_ptr<GraphDataTiling>(
                new ConvInputToOutputDirection(dynamic_cast<const ConvInputToOutputDirection&>(o)));
    } else if (o.getDirection() == Direction::OUTPUT_TO_INPUT) {
        return std::unique_ptr<GraphDataTiling>(
                new ConvOutputToInputDirection(dynamic_cast<const ConvOutputToInputDirection&>(o)));
    } else {
        IE_ASSERT(false) << "Unsupported direction";
    }
}

//
// Looks for the optimal tiling accordingly to the cost function. Modifies dimensions in dirTiling during search.
//
std::vector<TilingOption> HWConvolutionTilingSearcher::selectBetterTiling() const {
    const auto &env = CompileEnv::get();
    GraphDataTiling &dirTiling = *_dirTiling;
    FixedMaxHeap<TilingOption> tilingOptions(_maxTilingOptions);

    // TODO: estimate this numbers
    const int maxNumWidthTiles = 15;
    const int maxNumHeightTiles = 15;
    const int maxNumChannelTiles = _co._withPool ? 1 : 15;

    const auto outputTileInitial = dirTiling.getOutputTileDims();
    const auto inputTileInitial = dirTiling.getInputTileDims();

    auto minInputTileDimW = 64;
    auto minInputTileDimH = _co._kernelSizeY;
    if (_co._withPool) {
        minInputTileDimW *= 2;
        minInputTileDimH *= 2;
    }

    const DimValues &splitOver = dirTiling.splitOverTensorDims();
    const auto direction = dirTiling.getDirection();
    // split over Input tensor for the Channel dimension always
    for (int numChannelTiles = 1; numChannelTiles <= maxNumChannelTiles; numChannelTiles++) {
        const int tileSizeDimC = divUp(_co._inputDims[Dim::C], numChannelTiles);

        // here split and iterate either over input tensors or over output tensors depending on the direction.
        for (int numWidthTiles = 1; numWidthTiles <= maxNumWidthTiles; numWidthTiles++) {
            int tileSizeDimW = divUp(splitOver[Dim::W], numWidthTiles);

            //
            // Filter-out too small SoW input tiles when loops split input tensors.
            //

            if (numWidthTiles > 1 && direction == Direction::INPUT_TO_OUTPUT) {
                tileSizeDimW = divUp(tileSizeDimW, _co._kernelStride) * _co._kernelStride;

                if (tileSizeDimW < minInputTileDimW) {
                    break;
                }
            }

            for (int numHeightTiles = 1; numHeightTiles <= maxNumHeightTiles; numHeightTiles++) {
                int tileSizeDimH = divUp(splitOver[Dim::H], numHeightTiles);

                //
                // Filter-out too small SoH input tiles when loops split input tensors.
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

                dirTiling.setInputNOutputTileDimensions(tileSizeDimW, tileSizeDimH, tileSizeDimC);

                //
                // Limitations for Conv+Pool case.
                //

                if (_co._withPool) {
                    if (dirTiling.getOutputTileDims()[Dim::W] <= 2 ||
                        dirTiling.getOutputTileDims()[Dim::H] <= 2) {
                        break;
                    }
                }

                //
                // Check that tiling is valid.
                //

                // todo: check internal in/out hardcodes
                const auto heightTiles = calcHeightTiles(_co, dirTiling.getOutputTileDims(),
                                                         dirTiling.useCeil());
                const auto widthTiles = calcWidthTiles(_co, dirTiling.getOutputTileDims(), dirTiling.useCeil());

                if (heightTiles.empty()) {
                    continue;
                }
                if (widthTiles.empty()) {
                    break;
                }

                bool isOK = true;
                double solutionCost = 0.0;

                for (const auto &heightTile : heightTiles) {
                    for (const auto &widthTile : widthTiles) {
                        //
                        // Limitations for Conv+Pool case.
                        //

                        if (_co._withPool) {
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

                        auto tileInfo = splitHwConvIntoOutChannelsTiles(  // left asis, not new ver in new api
                                widthTile.inputWithJunk, heightTile.inputWithJunk, tileSizeDimC,
                                outputTileInitial[Dim::C],
                                _co._kernelSizeX, _co._kernelSizeY, _co._kernelStride);

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
                        fullOutputTileDims.set(Dim::C, outputTileInitial[Dim::C]);

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
                                            * outputTileInitial[Dim::C];
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
                                        * outputTileInitial[Dim::C];
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

                const int totalNumTiles = numWidthTiles * numHeightTiles * numChannelTiles;

                const TilingOption to =
                        {numWidthTiles, numHeightTiles, numChannelTiles, totalNumTiles, solutionCost};
                tilingOptions.push(to);

                // Skip smaller SoC tiling.
                break;
            }
        }
    }

    dirTiling.resetInputTileDims(inputTileInitial);
    dirTiling.resetOutputTileDims(outputTileInitial);

    return tilingOptions.sorted();
}

HWConvolutionTileLayoutCut HWConvolutionTilingSearcher::tileLayoutCut(const TilingOption &option) const {
    return HWConvolutionTileLayoutCut(*_dirTiling, option);
}

std::ostream& operator<<(std::ostream &o, const TilingOption &to) {
    o << "WHC: "
        << to.numWidthTiles << "x"
        << to.numHeightTiles << "x"
        << to.numChannelTiles
        << " Tot: " << to.totalNumTiles << " " << " cost: " << to.cost;

    return o;
}

// based on height of the tile for output tensor
SmallVector<HwPlaneTileInfo> calcHeightTiles(const ConvolutionOptions &_co,
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
        if (_co._withPool) {
            heightTiles = splitIntoPlaneTilesWithPool(
                    _co._inputDims[Dim::H],
                    _co._kernelSizeY,
                    _co._kernelStride,
                    _co._paddingTop,
                    outputTileDims[Dim::H]);
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
    }

    return heightTiles;
}

SmallVector<HwPlaneTileInfo> calcWidthTiles(const ConvolutionOptions &_co,
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
        if (_co._withPool) {
            widthTiles = splitIntoPlaneTilesWithPool(
                    _co._inputDims[Dim::W],
                    _co._kernelSizeX,
                    _co._kernelStride,
                    _co._paddingLeft,
                    outputTileDims[Dim::W]);
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
    }

    return widthTiles;
}

}  // namespace HWTilingNS

}  // namespace vpu
