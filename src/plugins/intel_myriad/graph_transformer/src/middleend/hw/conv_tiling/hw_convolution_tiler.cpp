// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <utility>
#include <vpu/middleend/hw/conv_tiling/hw_convolution_tiler.hpp>

namespace vpu {

namespace HWTilingNS {

bool operator<(const TilingOption& lhs, const TilingOption& rhs) {
    return lhs.cost < rhs.cost || (isDoubleEqual(lhs.cost, rhs.cost) && lhs.totalNumTiles < rhs.totalNumTiles);
}

static
void correctOutputPlaneSizeF(const ConvolutionOptions& convolutionOptions, bool _useCeil,
                             const DimValues& inputTileDims, DimValues& outputTileDims) {
    auto maxOutputWidth = calcOutputSize(
        inputTileDims[Dim::W],
        convolutionOptions._kernelSizeX,
        convolutionOptions._kernelStride,
        convolutionOptions._paddingLeft,
        convolutionOptions._paddingRight,
        _useCeil);

    if (convolutionOptions._withPool) {
        maxOutputWidth /= 2;
    }

    outputTileDims.set(Dim::W, std::min(outputTileDims[Dim::W], maxOutputWidth));

    auto maxOutputHeight = calcOutputSize(
        inputTileDims[Dim::H],
        convolutionOptions._kernelSizeY,
        convolutionOptions._kernelStride,
        convolutionOptions._paddingTop,
        convolutionOptions._paddingBottom,
        _useCeil);

    if (convolutionOptions._withPool) {
        maxOutputHeight /= 2;
    }

    outputTileDims.set(Dim::H, std::min(outputTileDims[Dim::H], maxOutputHeight));
}

// If output tile size appears too small to cover whole output
// tensor size with the given number of tiles, then we re-work
// the input tile size using the output-to-input direction.
static
void updateInputTileSize(int & inputTileSize,
                         int   numTiles,
                         int   outputSize,
                         int   kernelSize,
                         int   kernelStride,
                         int   padBefore,
                         int   padAfter,
                         bool  useCeil) {
    int outputTileSizeMin = divUp(outputSize, numTiles);
    int outputTileSize = calcOutputSize(inputTileSize,
                                        kernelSize,
                                        kernelStride,
                                        padBefore,
                                        padAfter,
                                        useCeil);
    if (outputTileSize < outputTileSizeMin) {
        inputTileSize = calcInputSize(outputTileSizeMin,
                                      kernelSize,
                                      kernelStride,
                                      padBefore,
                                      padAfter);
    }
}

class ConvInputToOutputDirection;
class ConvOutputToInputDirection;

// Input -> Output case
class ConvInputToOutputDirection : public GraphDataTiling {
public:
    explicit ConvInputToOutputDirection(const ConvolutionOptions& convolutionOptions) :
        GraphDataTiling(convolutionOptions, Direction::INPUT_TO_OUTPUT) {}
    ConvInputToOutputDirection(const ConvInputToOutputDirection&) = default;

    void initTileSizes() override {
        _useCeil = ceilNeeded();

        _inputTileDims.set(Dim::W, std::min(CNN_MAX_INPUT_WIDTH, _convolutionOptions._inputDims[Dim::W]));
        _inputTileDims.set(Dim::H, std::min(CNN_MAX_INPUT_HEIGHT, _convolutionOptions._inputDims[Dim::H]));
        _inputTileDims.set(Dim::C, std::min(CNN_MAX_INPUT_CHANNELS, _convolutionOptions._inputDims[Dim::C]));

        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::C, _convolutionOptions._outputDims[Dim::C]);

        correctOutputPlaneSize();
    }

    // Input -> Output case
    void setInputNOutputTileDimensions(int tileDimW, int tileDimH, int tileDimC) override {
        _inputTileDims.set(Dim::W, tileDimW);
        _inputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::C, tileDimC);

        correctOutputPlaneSize();
    }

    // Input -> Output case
    void applyTilingOption(const TilingOption& tilingOption) override {
        auto tileDimW = divUp(_convolutionOptions._inputDims[Dim::W], tilingOption.numWidthTiles);
        auto tileDimH = divUp(_convolutionOptions._inputDims[Dim::H], tilingOption.numHeightTiles);
        const auto tileDimC = divUp(_convolutionOptions._inputDims[Dim::C], tilingOption.numChannelTiles);

        if (tilingOption.numWidthTiles > 1) {
            tileDimW = divUp(tileDimW, _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;
        }
        if (tilingOption.numHeightTiles > 1) {
            tileDimH = divUp(tileDimH, _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;

            updateInputTileSize(tileDimH,
                                tilingOption.numHeightTiles,
                                _convolutionOptions._outputDims[Dim::H],
                                _convolutionOptions._kernelSizeY,
                                _convolutionOptions._kernelStride,
                                _convolutionOptions._paddingBottom,
                                _convolutionOptions._paddingTop,
                                false);  // do not use ceil
        }

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
        correctOutputPlaneSizeF(
            _convolutionOptions,
            _useCeil,
            _inputTileDims,
            _outputTileDims);
    }

    const DimValues& splitOverTensorDims() override {
        return _convolutionOptions._inputDims;
    }

    bool patternMatching() override {
        return GraphDataTiling::patternMatching();
    }
};

// Output -> Input case
class ConvOutputToInputDirection: public GraphDataTiling {
public:
    explicit ConvOutputToInputDirection(const ConvolutionOptions& convolutionOptions) :
        GraphDataTiling(convolutionOptions, Direction::OUTPUT_TO_INPUT) {}
    ConvOutputToInputDirection(const ConvOutputToInputDirection&) = default;

    void initTileSizes() override {
        _useCeil = false;   // no ceiling needed for calculating input sizes from output sizes

        _outputTileDims.set(Dim::W, std::min(CNN_MAX_INPUT_WIDTH, _convolutionOptions._outputDims[Dim::W]));
        _outputTileDims.set(Dim::H, std::min(CNN_MAX_INPUT_HEIGHT, _convolutionOptions._outputDims[Dim::H]));
        _outputTileDims.set(Dim::C, _convolutionOptions._outputDims[Dim::C]);

        _inputTileDims.set(Dim::W, std::min(CNN_MAX_INPUT_WIDTH, _convolutionOptions._inputDims[Dim::W]));
        _inputTileDims.set(Dim::H, std::min(CNN_MAX_INPUT_HEIGHT, _convolutionOptions._inputDims[Dim::H]));
        _inputTileDims.set(Dim::C, std::min(CNN_MAX_INPUT_CHANNELS, _convolutionOptions._inputDims[Dim::C]));

        correctInputPlaneSize();
    }

    // Output -> Input case
    void setInputNOutputTileDimensions(int tileDimW, int tileDimH, int tileDimC) override {
        _outputTileDims.set(Dim::W, tileDimW);
        _outputTileDims.set(Dim::H, tileDimH);
        _outputTileDims.set(Dim::C, tileDimC);

        correctInputPlaneSize();
    }

    // Output -> Input case
    void applyTilingOption(const TilingOption& tilingOption) override {
        const auto tileDimW = divUp(_convolutionOptions._outputDims[Dim::W], tilingOption.numWidthTiles);
        const auto tileDimH = divUp(_convolutionOptions._outputDims[Dim::H], tilingOption.numHeightTiles);
        // split only input tensor over C dim
        const auto tileDimC = divUp(_convolutionOptions._inputDims[Dim::C], tilingOption.numChannelTiles);

        _outputTileDims.set(Dim::W, tileDimW);
        _outputTileDims.set(Dim::H, tileDimH);
        _inputTileDims.set(Dim::C, tileDimC);

        correctInputPlaneSize();
    }

    void correctPlaneSize() override {
        correctInputPlaneSize();
    }

    void correctPlaneSizeAfterPatternMatching() override {
        correctOutputPlaneSizeF(
            _convolutionOptions,
            _useCeil,
            _inputTileDims,
            _outputTileDims);
    }

    void correctInputPlaneSize() {
        auto maxInputWidth = calcInputSize(
            _outputTileDims[Dim::W],
            _convolutionOptions._kernelSizeX,
            _convolutionOptions._kernelStride,
            _convolutionOptions._paddingLeft,
            _convolutionOptions._paddingRight);

        if (_convolutionOptions._withPool) {
            maxInputWidth *= 2;
        }

        _inputTileDims.set(Dim::W, std::min(_inputTileDims[Dim::W], maxInputWidth));

        auto maxInputHeight = calcInputSize(
            _outputTileDims[Dim::H],
            _convolutionOptions._kernelSizeY,
            _convolutionOptions._kernelStride,
            _convolutionOptions._paddingTop,
            _convolutionOptions._paddingBottom);

        if (_convolutionOptions._withPool) {
            maxInputHeight *= 2;
        }

        _inputTileDims.set(Dim::H, std::min(_inputTileDims[Dim::H], maxInputHeight));
    }

    const DimValues& splitOverTensorDims() override {
        return _convolutionOptions._outputDims;
    }

    bool patternMatching() override {
        return false;
    }
};

HWConvolutionTiler::HWConvolutionTiler(const ConvolutionOptions& convolutionOptions, const Direction& direction,
                                       std::size_t maxTilingOptions) :
    _convolutionOptions(std::move(convolutionOptions)),
    _searcher(_convolutionOptions, direction, maxTilingOptions) {
    _tilingPossible = tileForHW();
}

bool HWConvolutionTiler::tileForHW() {
    const auto& tilingOptions = _searcher.tilingOptions();
    if (tilingOptions.empty()) {
        return false;
    }

    for (const TilingOption& tilingOption : tilingOptions) {
        const auto& tileLayoutCut = _searcher.tileLayoutCut(tilingOption);
        if (tileLayoutCut.tileCutPossible()) {
            _hwTilings.push_back(tileLayoutCut.hwTiling());
        }
    }

    return !_hwTilings.empty();
}

bool GraphDataTiling::ceilNeeded() const {
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

    const auto outWidthWithOutCeil = (tempX + _convolutionOptions._kernelStride) / _convolutionOptions._kernelStride;
    const auto outHeightWithOutCeil = (tempY + _convolutionOptions._kernelStride) / _convolutionOptions._kernelStride;

    const auto outWidthWithCeil = static_cast<int>(
        std::ceil(static_cast<double>(tempX) / _convolutionOptions._kernelStride + 1));
    const auto outHeightWithCeil = static_cast<int>(
        std::ceil(static_cast<double>(tempY) / _convolutionOptions._kernelStride + 1));

    if ((_convolutionOptions._origOutputDims[Dim::W] != outWidthWithCeil) &&
        (_convolutionOptions._origOutputDims[Dim::W] != outWidthWithOutCeil)) {
        VPU_THROW_EXCEPTION << "Internal error: Output in " << _convolutionOptions._stageName
                            << " has incorrect width dimension. Expected: " << outWidthWithCeil << " or "
                            << outWidthWithOutCeil << " Actual: " << _convolutionOptions._origOutputDims[Dim::W];
    }

    if ((_convolutionOptions._origOutputDims[Dim::H] != outHeightWithCeil) &&
        (_convolutionOptions._origOutputDims[Dim::H] != outHeightWithOutCeil)) {
        VPU_THROW_EXCEPTION << "Internal error: Output in " << _convolutionOptions._stageName
                            << " has incorrect height dimension. Expected: " << outHeightWithCeil << " or "
                            << outHeightWithOutCeil << " Actual: " << _convolutionOptions._origOutputDims[Dim::H];
    }

    return ((_convolutionOptions._origOutputDims[Dim::W] == outWidthWithCeil) ||
            (_convolutionOptions._origOutputDims[Dim::H] == outHeightWithCeil));
}

bool GraphDataTiling::patternMatching() {
    // All optimizations below are for MiryadX code with 2 threads, so at least 9 slices is required.
    // TODO: check 1-thread perfomance and replace with exact equality check.
    if (CompileEnv::get().resources.numCMXSlices < 9) {
        return false;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._kernelSizeX == 3 &&
         _convolutionOptions._kernelSizeY == 3 &&
         _convolutionOptions._paddingLeft == 1 &&
         _convolutionOptions._paddingRight == 1 &&
         _convolutionOptions._paddingTop == 1 &&
         _convolutionOptions._paddingBottom == 1 &&
         _convolutionOptions._kernelStride == 1 &&
         _convolutionOptions._inputDims[Dim::C] == 512 &&
         _convolutionOptions._inputDims[Dim::H] == 28 &&
         _convolutionOptions._inputDims[Dim::W] == 28 &&
         _convolutionOptions._outputDims[Dim::C] == 512) {
        _inputTileDims.set(Dim::H, 28);
        _inputTileDims.set(Dim::C, 172);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._kernelSizeX == 3 &&
         _convolutionOptions._kernelSizeY == 3 &&
         _convolutionOptions._paddingLeft == 1 &&
         _convolutionOptions._paddingRight == 1 &&
         _convolutionOptions._paddingTop == 1 &&
         _convolutionOptions._paddingBottom == 1 &&
         _convolutionOptions._kernelStride == 1 &&
         _convolutionOptions._inputDims[Dim::C] == 256 &&
         _convolutionOptions._inputDims[Dim::H] == 56 &&
         _convolutionOptions._inputDims[Dim::W] == 56 &&
         _convolutionOptions._outputDims[Dim::C] == 256) {
        _inputTileDims.set(Dim::H, 30);
        _inputTileDims.set(Dim::C, 128);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._kernelSizeX == 3 &&
         _convolutionOptions._kernelSizeY == 3 &&
         _convolutionOptions._paddingLeft == 1 &&
         _convolutionOptions._paddingRight == 1 &&
         _convolutionOptions._paddingTop == 1 &&
         _convolutionOptions._paddingBottom == 1 &&
         _convolutionOptions._kernelStride == 1 &&
         _convolutionOptions._inputDims[Dim::C] == 64 &&
         _convolutionOptions._inputDims[Dim::H] == 224 &&
         _convolutionOptions._inputDims[Dim::W] == 224 &&
         _convolutionOptions._outputDims[Dim::C] == 64) {
        _inputTileDims.set(Dim::H, 82);
        _inputTileDims.set(Dim::W, 82);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (_convolutionOptions._inputDims[Dim::C] == 512 &&
        _convolutionOptions._inputDims[Dim::H] == 7 &&
        _convolutionOptions._inputDims[Dim::W] == 7 &&
        _convolutionOptions._outputDims[Dim::C] == 4096) {
        _inputTileDims.set(Dim::C, 64);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._kernelSizeX == 3 &&
         _convolutionOptions._kernelSizeY == 3 &&
         _convolutionOptions._paddingLeft == 1 &&
         _convolutionOptions._paddingRight == 1 &&
         _convolutionOptions._paddingTop == 1 &&
         _convolutionOptions._paddingBottom == 1 &&
         _convolutionOptions._kernelStride == 1 &&
         _convolutionOptions._inputDims[Dim::C] == 128 &&
         _convolutionOptions._inputDims[Dim::H] == 112 &&
         _convolutionOptions._inputDims[Dim::W] == 112 &&
         _convolutionOptions._outputDims[Dim::C] == 128) {
        _inputTileDims.set(Dim::H, 32);
        _inputTileDims.set(Dim::W, 112);
        _inputTileDims.set(Dim::C, 32);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (_convolutionOptions._inputDims[Dim::C] == 1088 &&
        _convolutionOptions._inputDims[Dim::H] == 17 &&
        _convolutionOptions._inputDims[Dim::W] == 17 &&
        (_convolutionOptions._outputDims[Dim::C] == 128 || _convolutionOptions._outputDims[Dim::C] == 192)) {
        _inputTileDims.set(Dim::H, 17);
        _inputTileDims.set(Dim::C, 544);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (_convolutionOptions._inputDims[Dim::C] == 1024 &&
        _convolutionOptions._inputDims[Dim::H] == 17 &&
        _convolutionOptions._inputDims[Dim::W] == 17 &&
        _convolutionOptions._outputDims[Dim::C] == 384) {
        _inputTileDims.set(Dim::H, 17);
        _inputTileDims.set(Dim::C, 512);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._kernelSizeX == 3 &&
         _convolutionOptions._kernelSizeY == 3 &&
         _convolutionOptions._paddingLeft == 0 &&
         _convolutionOptions._paddingRight == 0 &&
         _convolutionOptions._paddingTop == 0 &&
         _convolutionOptions._paddingBottom == 0 &&
         _convolutionOptions._kernelStride == 2 &&
         _convolutionOptions._inputDims[Dim::C] == 384 &&
         _convolutionOptions._inputDims[Dim::H] == 35 &&
         _convolutionOptions._inputDims[Dim::W] == 35 &&
         _convolutionOptions._outputDims[Dim::C] == 384) {
        _inputTileDims.set(Dim::C, 194);
        _inputTileDims.set(Dim::H, 35);
        _inputTileDims.set(Dim::W, 35);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (_convolutionOptions._inputDims[Dim::C] == 192 &&
        _convolutionOptions._inputDims[Dim::H] == 71 &&
        _convolutionOptions._inputDims[Dim::W] == 71 &&
        _convolutionOptions._outputDims[Dim::H] == 35) {
        _inputTileDims.set(Dim::W, 71);
        _inputTileDims.set(Dim::C, 96);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._inputDims[Dim::C] == 256 &&
         _convolutionOptions._inputDims[Dim::H] == 128 &&
         _convolutionOptions._inputDims[Dim::W] == 128 &&
         _convolutionOptions._outputDims[Dim::C] == 256) {
        _inputTileDims.set(Dim::W, 128);
        _inputTileDims.set(Dim::H, 15);
        _inputTileDims.set(Dim::C, 64);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._inputDims[Dim::C] == 512 &&
         _convolutionOptions._inputDims[Dim::H] == 64 &&
         _convolutionOptions._inputDims[Dim::W] == 64 &&
         _convolutionOptions._outputDims[Dim::C] == 512) {
        _inputTileDims.set(Dim::W, 64);
        _inputTileDims.set(Dim::H, 10);
        _inputTileDims.set(Dim::C, 128);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._kernelSizeX == 1 &&
         _convolutionOptions._kernelSizeY == 1 &&
         _convolutionOptions._paddingLeft == 0 &&
         _convolutionOptions._paddingRight == 0 &&
         _convolutionOptions._paddingTop == 0 &&
         _convolutionOptions._paddingBottom == 0 &&
         _convolutionOptions._kernelStride == 1 &&
         _convolutionOptions._inputDims[Dim::C] == 384 &&
         _convolutionOptions._inputDims[Dim::H] == 56 &&
         _convolutionOptions._inputDims[Dim::W] == 56 &&
         _convolutionOptions._outputDims[Dim::C] == 64) {
        _inputTileDims.set(Dim::C, 384);
        _inputTileDims.set(Dim::H, 56);
        _inputTileDims.set(Dim::W, 20);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._kernelSizeX == 1 &&
         _convolutionOptions._kernelSizeY == 1 &&
         _convolutionOptions._paddingLeft == 0 &&
         _convolutionOptions._paddingRight == 0 &&
         _convolutionOptions._paddingTop == 0 &&
         _convolutionOptions._paddingBottom == 0 &&
         _convolutionOptions._kernelStride == 1 &&
         _convolutionOptions._inputDims[Dim::C] == 2112 &&
         _convolutionOptions._inputDims[Dim::H] == 14 &&
         _convolutionOptions._inputDims[Dim::W] == 14 &&
         _convolutionOptions._outputDims[Dim::C] == 1056) {
        _inputTileDims.set(Dim::C, 556);
        _inputTileDims.set(Dim::H, 14);
        _inputTileDims.set(Dim::W, 14);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._kernelSizeX == 3 &&
         _convolutionOptions._kernelSizeY == 3 &&
         _convolutionOptions._paddingLeft == 1 &&
         _convolutionOptions._paddingRight == 1 &&
         _convolutionOptions._paddingTop == 1 &&
         _convolutionOptions._paddingBottom == 1 &&
         _convolutionOptions._kernelStride == 2 &&
         _convolutionOptions._inputDims[Dim::C] == 256 &&
         _convolutionOptions._inputDims[Dim::H] == 52 &&
         _convolutionOptions._inputDims[Dim::W] == 52 &&
         _convolutionOptions._outputDims[Dim::C] == 512) {
        _inputTileDims.set(Dim::C, 128);
        _inputTileDims.set(Dim::H, 52);
        _inputTileDims.set(Dim::W, 52);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    if (!_convolutionOptions._withPool &&
         _convolutionOptions._kernelSizeX == 3 &&
         _convolutionOptions._kernelSizeY == 3 &&
         _convolutionOptions._paddingLeft == 1 &&
         _convolutionOptions._paddingRight == 1 &&
         _convolutionOptions._paddingTop == 1 &&
         _convolutionOptions._paddingBottom == 1 &&
         _convolutionOptions._kernelStride == 1 &&
         _convolutionOptions._inputDims[Dim::C] == 256 &&
         _convolutionOptions._inputDims[Dim::H] == 23 &&
         _convolutionOptions._inputDims[Dim::W] == 23 &&
         _convolutionOptions._outputDims[Dim::C] == 640) {
        _inputTileDims.set(Dim::C, 256);
        _inputTileDims.set(Dim::H, 14);
        _inputTileDims.set(Dim::W, 23);
        _outputTileDims.set(Dim::H, _convolutionOptions._outputDims[Dim::H]);
        _outputTileDims.set(Dim::W, _convolutionOptions._outputDims[Dim::W]);

        return true;
    }

    return false;
}

std::unique_ptr<GraphDataTiling> ConvGraphDataTilingFactory::makeDirTiling(const ConvolutionOptions& convolutionOptions,
                                                                           const Direction& direction) {
    if (direction == Direction::INPUT_TO_OUTPUT) {
        return std::unique_ptr<GraphDataTiling>(new ConvInputToOutputDirection(convolutionOptions));
    } else if (direction == Direction::OUTPUT_TO_INPUT) {
        return std::unique_ptr<GraphDataTiling>(new ConvOutputToInputDirection(convolutionOptions));
    }
    IE_THROW() << "Unsupported direction";
}

std::unique_ptr<GraphDataTiling> ConvGraphDataTilingFactory::makeDirTiling(const GraphDataTiling& graphDataTiling) {
    if (graphDataTiling.getDirection() == Direction::INPUT_TO_OUTPUT) {
        return std::unique_ptr<GraphDataTiling>(
            new ConvInputToOutputDirection(dynamic_cast<const ConvInputToOutputDirection&>(graphDataTiling)));
    } else if (graphDataTiling.getDirection() == Direction::OUTPUT_TO_INPUT) {
        return std::unique_ptr<GraphDataTiling>(
            new ConvOutputToInputDirection(dynamic_cast<const ConvOutputToInputDirection&>(graphDataTiling)));
    }
    IE_THROW() << "Unsupported direction";
}

//
// Looks for the optimal tiling accordingly to the cost function. Modifies dimensions in dirTiling during search.
//
std::vector<TilingOption> HWConvolutionTilingSearcher::selectBetterTiling() const {
    const auto& env = CompileEnv::get();

    auto& dirTiling = *_dirTiling;
    FixedMaxHeap<TilingOption> tilingOptions(_maxTilingOptions);

    // TODO: estimate this numbers
    const int maxNumWidthTiles = 15;
    const int maxNumHeightTiles = 15;
    const int maxNumChannelTiles = _convolutionOptions._withPool ? 1 : 15;

    const auto outputTileInitial = dirTiling.getOutputTileDims();
    const auto inputTileInitial = dirTiling.getInputTileDims();

    const int maxInputTileDimW = 2048;
    const int maxInputTileDimH = 2048;
    const int maxInputTileDimC = 2048;

    auto minInputTileDimW = 64;
    auto minInputTileDimH = _convolutionOptions._kernelSizeY;
    if (_convolutionOptions._withPool) {
        minInputTileDimW *= 2;
        minInputTileDimH *= 2;
    }

    const auto& splitOver = dirTiling.splitOverTensorDims();
    const auto direction = dirTiling.getDirection();
    const auto cmxLimit = env.resources.tilingCMXLimit;

    // split over Input tensor for the Channel dimension always
    for (int numChannelTiles = 1; numChannelTiles <= maxNumChannelTiles; numChannelTiles++) {
        const int tileSizeDimC = divUp(_convolutionOptions._inputDims[Dim::C], numChannelTiles);

        if (tileSizeDimC > maxInputTileDimC)
            continue;
        // here split and iterate either over input tensors or over output tensors depending on the direction.
        for (int numWidthTiles = 1; numWidthTiles <= maxNumWidthTiles; numWidthTiles++) {
            int tileSizeDimW = divUp(splitOver[Dim::W], numWidthTiles);

            if (tileSizeDimW > maxInputTileDimW)
                continue;

            //
            // Filter-out too small SoW input tiles when loops split input tensors.
            //

            if (numWidthTiles > 1 && direction == Direction::INPUT_TO_OUTPUT) {
                tileSizeDimW = divUp(tileSizeDimW,
                                     _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;

                if (tileSizeDimW < minInputTileDimW) {
                    break;
                }
            }

            for (int numHeightTiles = 1; numHeightTiles <= maxNumHeightTiles; numHeightTiles++) {
                int tileSizeDimH = divUp(splitOver[Dim::H], numHeightTiles);

                if (tileSizeDimH > maxInputTileDimH)
                    continue;

                //
                // Filter-out too small SoH input tiles when loops split input tensors.
                //
                if (numHeightTiles > 1 && direction == Direction::INPUT_TO_OUTPUT) {
                    tileSizeDimH = divUp(tileSizeDimH,
                                         _convolutionOptions._kernelStride) * _convolutionOptions._kernelStride;

                    updateInputTileSize(tileSizeDimH,
                                        numHeightTiles,
                                        _convolutionOptions._outputDims[Dim::H],
                                        _convolutionOptions._kernelSizeY,
                                        _convolutionOptions._kernelStride,
                                        _convolutionOptions._paddingBottom,
                                        _convolutionOptions._paddingTop,
                                        false);  // do not use ceil

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

                if (_convolutionOptions._withPool) {
                    if (dirTiling.getOutputTileDims()[Dim::W] <= 2 || dirTiling.getOutputTileDims()[Dim::H] <= 2) {
                        break;
                    }
                }

                //
                // Check that tiling is valid.
                //

                // TODO: check internal in/out hardcodes
                const auto heightTiles = calcHeightTiles(
                    _convolutionOptions, dirTiling.getOutputTileDims(),
                    dirTiling.useCeil());
                const auto widthTiles = calcWidthTiles(
                    _convolutionOptions, dirTiling.getOutputTileDims(),
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
                        // Limitations for Conv+Pool case.
                        //

                        if (_convolutionOptions._withPool) {
                            if (widthTile.inputWithJunk % 2 != 0 || heightTile.inputWithJunk % 2 != 0 ||
                                widthTile.outputWithJunk % 2 != 0 || widthTile.outputWithJunk <= 2 ||
                                heightTile.outputWithJunk <= 2 ||
                                // this restrictions come from restrictions on HW tile sizes in case of Conv+Pool:
                                (tileSizeDimC <= 128 && tileSizeDimC > 112 && widthTile.inputWithJunk > 72) ||
                                (tileSizeDimC <= 112 && tileSizeDimC > 96  && widthTile.inputWithJunk > 80) ||
                                (tileSizeDimC <= 96  && tileSizeDimC > 80  && widthTile.inputWithJunk > 96) ||
                                (tileSizeDimC <= 80  && tileSizeDimC > 64  && widthTile.inputWithJunk > 112) ||
                                (tileSizeDimC <= 64  && tileSizeDimC > 48  && widthTile.inputWithJunk > 144) ||
                                (tileSizeDimC <= 48  && tileSizeDimC > 32  && widthTile.inputWithJunk > 192) ||
                                (tileSizeDimC <= 32  && tileSizeDimC > 16  && widthTile.inputWithJunk > 288)) {
                                isOK = false;
                                break;
                            }
                        }

                        //
                        // Can use this tile.
                        //

                        const auto tileInfo = splitHwConvIntoOutChannelsTiles(  // left asis, not new ver in new api
                            widthTile.inputWithJunk, heightTile.inputWithJunk, tileSizeDimC,
                            outputTileInitial[Dim::C],
                            _convolutionOptions._kernelSizeX,
                            _convolutionOptions._kernelSizeY,
                            _convolutionOptions._kernelStride);

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
                        if (calculateHwBufferSize(fullOutputTileDims) > cmxLimit) {
                            isOK = false;
                            break;
                        }

                        //
                        // Calc tile cost.
                        //

                        solutionCost += tileInfo.cost * numChannelTiles;

                        // Alignment for output
                        if ((widthTile.outputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                            solutionCost += static_cast<double>(widthTile.outputWithJunk)
                                            * heightTile.outputWithJunk
                                            * outputTileInitial[Dim::C];
                        }

                        // Alignment for input
                        if ((widthTile.inputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                            solutionCost += static_cast<double>(widthTile.inputWithJunk)
                                            * heightTile.inputWithJunk
                                            * tileInfo.extendedInputDimC;
                        }

                        // SoC overhead
                        solutionCost += static_cast<double>((numChannelTiles - 1))
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
                tilingOptions.push({numWidthTiles, numHeightTiles, numChannelTiles, totalNumTiles, solutionCost});

                // Skip smaller SoC tiling.
                break;
            }
        }
    }

    dirTiling.resetInputTileDims(inputTileInitial);
    dirTiling.resetOutputTileDims(outputTileInitial);

    return tilingOptions.sorted();
}

HWConvolutionTileLayoutCut HWConvolutionTilingSearcher::tileLayoutCut(const TilingOption& option) const {
    return HWConvolutionTileLayoutCut(*_dirTiling, option);
}

std::ostream& operator<<(std::ostream& stream, const TilingOption& tilingOption) {
    stream << "WHC: "
           << tilingOption.numWidthTiles << "x"
           << tilingOption.numHeightTiles << "x"
           << tilingOption.numChannelTiles
           << " Tot: " << tilingOption.totalNumTiles << " " << " cost: " << tilingOption.cost;

    return stream;
}

// based on height of the tile for output tensor
SmallVector<HwPlaneTileInfo> calcHeightTiles(const ConvolutionOptions& convolutionOptions,
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
        if (convolutionOptions._withPool) {
            heightTiles = splitIntoPlaneTilesWithPool(
                convolutionOptions._inputDims[Dim::H],
                convolutionOptions._kernelSizeY,
                convolutionOptions._kernelStride,
                convolutionOptions._paddingTop,
                outputTileDims[Dim::H]);
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
    }

    return heightTiles;
}

SmallVector<HwPlaneTileInfo> calcWidthTiles(const ConvolutionOptions& convolutionOptions,
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
        if (convolutionOptions._withPool) {
            widthTiles = splitIntoPlaneTilesWithPool(
                convolutionOptions._inputDims[Dim::W],
                convolutionOptions._kernelSizeX,
                convolutionOptions._kernelStride,
                convolutionOptions._paddingLeft,
                outputTileDims[Dim::W]);
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
    }

    return widthTiles;
}

}  // namespace HWTilingNS

}  // namespace vpu
