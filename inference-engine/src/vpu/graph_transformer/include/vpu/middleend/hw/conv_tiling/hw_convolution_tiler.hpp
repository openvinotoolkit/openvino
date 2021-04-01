// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <memory>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <vpu/model/data_desc.hpp>
#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/utils/heap.hpp>

namespace vpu {

namespace HWTilingNS {

struct ConvolutionOptions final {
    const std::string _stageName;

    const DimValues _inputDims;
    const DimValues _outputDims;
    const DimValues _origOutputDims;

    const int _kernelSizeX;
    const int _kernelSizeY;
    const int _kernelStride;
    const int _paddingLeft;
    const int _paddingRight;
    const int _paddingTop;
    const int _paddingBottom;

    const bool _withPool;

public:
    ConvolutionOptions(std::string stageName, const DimValues& inputDims, const DimValues& outputDims,
                       const DimValues& origOutputDims, int kernelSizeX, int kernelSizeY,
                       int kernelStride, int paddingLeft, int paddingRight,
                       int paddingTop, int paddingBottom, bool withPool)
            : _stageName(std::move(stageName)), _inputDims(inputDims), _outputDims(outputDims),
              _origOutputDims(origOutputDims), _kernelSizeX(kernelSizeX), _kernelSizeY(kernelSizeY),
              _kernelStride(kernelStride), _paddingLeft(paddingLeft), _paddingRight(paddingRight),
              _paddingTop(paddingTop), _paddingBottom(paddingBottom), _withPool(withPool) {}
};

struct TilingOption final {
    int numWidthTiles;
    int numHeightTiles;
    int numChannelTiles;
    int totalNumTiles;
    double cost;
};

bool operator<(const TilingOption& lhs, const TilingOption& rhs);

std::ostream& operator<<(std::ostream& stream, const TilingOption& tilingOption);

enum class Direction {
    INPUT_TO_OUTPUT = 0, OUTPUT_TO_INPUT = 1
};

// Tensors can be split going either from input to output or vice versa
class GraphDataTiling {
public:
    GraphDataTiling() = delete;

    virtual ~GraphDataTiling() = default;
    GraphDataTiling(const GraphDataTiling&) = default;

    GraphDataTiling(const ConvolutionOptions& convolutionOptions, const Direction& direction) :
        _convolutionOptions(convolutionOptions), _direction(direction) {}

    const DimValues& getInputTileDims() const { return _inputTileDims; }

    const DimValues& getOutputTileDims() const { return _outputTileDims; }

    DimValues& getInputTileDims() { return _inputTileDims; }

    DimValues& getOutputTileDims() { return _outputTileDims; }

    void resetInputTileDims(const DimValues& dimValues) { _inputTileDims = dimValues; }

    void resetOutputTileDims(const DimValues& dimValues) { _outputTileDims = dimValues; }

    virtual void initTileSizes() = 0;

    virtual void applyTilingOption(const TilingOption& tilingOption) = 0;

    virtual void setInputNOutputTileDimensions(int tileDimW, int tileDimH, int tileDimC) = 0;

    virtual void correctPlaneSize() = 0;

    virtual void correctPlaneSizeAfterPatternMatching() = 0;

    virtual const DimValues& splitOverTensorDims() = 0;

    virtual bool patternMatching();

    bool useCeil() const { return _useCeil; }

    Direction getDirection() const {
        return _direction;
    }

    const ConvolutionOptions& convolutionOptions() const { return _convolutionOptions; }

protected:
    const ConvolutionOptions& _convolutionOptions;
    // size of every tile for input tensor in each dimension
    DimValues _inputTileDims;
    // size of every tile for output tensor in each dimension
    DimValues _outputTileDims;
    bool _useCeil = false;
    const enum Direction _direction;

    bool ceilNeeded() const;
};

class ConvGraphDataTilingFactory final {
public:
    static std::unique_ptr<GraphDataTiling> makeDirTiling(const ConvolutionOptions& convolutionOptions,
                                                          const Direction& direction);
    static std::unique_ptr<GraphDataTiling> makeDirTiling(const GraphDataTiling& graphDataTiling);
};

class HWConvolutionTileLayoutCut;

// iterates over all the tiling options and chooses few with minimal cost
class HWConvolutionTilingSearcher {
public:
    HWConvolutionTilingSearcher() = delete;

    HWConvolutionTilingSearcher(const HWConvolutionTilingSearcher& other):
        _convolutionOptions(other._convolutionOptions),
        _maxTilingOptions(other._maxTilingOptions),
        _dirTiling(ConvGraphDataTilingFactory::makeDirTiling(*other._dirTiling)),
        _tilingOptions(other._tilingOptions) {}
    HWConvolutionTilingSearcher(ConvolutionOptions convolutionOptions, const Direction& direction,
                                std::size_t maxTilingOptions) :
        _convolutionOptions(std::move(convolutionOptions)),
        _dirTiling(ConvGraphDataTilingFactory::makeDirTiling(_convolutionOptions, direction)),
        _maxTilingOptions(maxTilingOptions) {
            IE_ASSERT(maxTilingOptions > 0);
            _dirTiling->initTileSizes();
            _tilingOptions = selectBetterTiling();
        }

    const std::vector<TilingOption>& tilingOptions() const {
        return _tilingOptions;
    }

    const ConvolutionOptions& convolutionOptions() const { return _convolutionOptions; }

    HWConvolutionTileLayoutCut tileLayoutCut(const TilingOption& option) const;

private:
    std::vector<TilingOption> selectBetterTiling() const;

    const ConvolutionOptions _convolutionOptions;
    const std::size_t _maxTilingOptions;
    const std::unique_ptr<GraphDataTiling> _dirTiling;
    std::vector<TilingOption> _tilingOptions;
};

// Search for tiling options and applies them to prepare hw tilings
class HWConvolutionTiler final {
public:
    HWConvolutionTiler() = delete;
    HWConvolutionTiler(const HWConvolutionTiler&) = default;
    HWConvolutionTiler(ConvolutionOptions convolutionOptions, const Direction& direction, std::size_t maxTilingOptions);


    bool isTilingPossible() const {
        return _tilingPossible;
    }

    bool withPool() const {
        return _convolutionOptions._withPool;
    }

    const std::vector<HwConvTilingPtr>& getHwTilings() const {
        return _hwTilings;
    }

private:
    bool tileForHW();

    const ConvolutionOptions _convolutionOptions;
    std::vector<HwConvTilingPtr> _hwTilings;
    bool _tilingPossible;
    const HWConvolutionTilingSearcher _searcher;
};

SmallVector<HwPlaneTileInfo> calcHeightTiles(const ConvolutionOptions& convolutionOptions,
                                             const DimValues& outputTileDims, bool useCeil);
SmallVector<HwPlaneTileInfo> calcWidthTiles(const ConvolutionOptions& convolutionOptions,
                                            const DimValues& outputTileDims, bool useCeil);

// Based on chosen { inputTileDims, outputTileDims } constructs plane's tiling structure;
// (same for both input and output, contains only number of tiles in each dimension)
class HWConvolutionTileLayoutCut {
public:
    HWConvolutionTileLayoutCut() = delete;
    HWConvolutionTileLayoutCut(const HWConvolutionTileLayoutCut&) = default;
    HWConvolutionTileLayoutCut(HWConvolutionTileLayoutCut&& other) noexcept :
        _convolutionOptions(other._convolutionOptions), _dirTiling(other._dirTiling) {
        _hwTiling = std::move(other._hwTiling);
        _tileCutPossible = other.tileCutPossible();
    }

    HWConvolutionTileLayoutCut(GraphDataTiling& dirTiling, const TilingOption& tilingOption) :
        _dirTiling(dirTiling),
        _convolutionOptions(dirTiling.convolutionOptions()), _hwTiling(std::make_shared<HwConvTiling>()) {
        dirTiling.applyTilingOption(tilingOption);

        if (dirTiling.patternMatching()) {
            dirTiling.correctPlaneSizeAfterPatternMatching();
        }

        // Merged Pooling and SoC can't be used together.
        if (_convolutionOptions._withPool) {
            IE_ASSERT(!hasSoC(dirTiling));
        }

        const auto& heightTiles = calcHeightTiles(
            _convolutionOptions,
            dirTiling.getOutputTileDims(),
            dirTiling.useCeil());
        const auto& widthTiles = calcWidthTiles(
            _convolutionOptions,
            dirTiling.getOutputTileDims(),
            dirTiling.useCeil());

        _tileCutPossible = createTiles(
            heightTiles,
            widthTiles,
            dirTiling.getInputTileDims(),
            dirTiling.getOutputTileDims());
    }

    bool tileCutPossible() const { return _tileCutPossible; }

    HwConvTilingPtr hwTiling() const {
        IE_ASSERT(_tileCutPossible);
        return _hwTiling;
    }

private:
    bool createTiles(const SmallVector<HwPlaneTileInfo>& heightTiles,
                     const SmallVector<HwPlaneTileInfo>& widthTiles,
                     const DimValues& inputTileDims, const DimValues& outputTileDims) const {
        IE_ASSERT(!heightTiles.empty());
        IE_ASSERT(!widthTiles.empty());

        _hwTiling->sohTiles = heightTiles.size();
        _hwTiling->sowTiles = widthTiles.size();
        _hwTiling->socTiles = divUp(_convolutionOptions._inputDims[Dim::C], inputTileDims[Dim::C]);

        for (int sohInd = 0; sohInd < _hwTiling->sohTiles; ++sohInd) {
            const auto& heightTileInfo = heightTiles[sohInd];

            for (int sowInd = 0; sowInd < _hwTiling->sowTiles; ++sowInd) {
                const auto& widthTileInfo = widthTiles[sowInd];

                auto planeTile = std::make_shared<HwConvPlaneTile>();
                planeTile->parent = _hwTiling;

                planeTile->sohInd = sohInd;
                planeTile->sowInd = sowInd;

                planeTile->heightInfo = heightTileInfo;
                planeTile->widthInfo = widthTileInfo;

                for (int socInd = 0; socInd < _hwTiling->socTiles; ++socInd) {
                    auto channelTile = std::make_shared<HwConvChannelTile>();
                    channelTile->parent = planeTile;

                    channelTile->socInd = socInd;

                    channelTile->finalTiles = splitHwConvIntoOutChannelsTiles(
                        widthTileInfo.inputWithJunk,
                        heightTileInfo.inputWithJunk,
                        inputTileDims[Dim::C],
                        outputTileDims[Dim::C],
                        _convolutionOptions._kernelSizeX,
                        _convolutionOptions._kernelSizeY,
                        _convolutionOptions._kernelStride);

                    if (channelTile->finalTiles.numDescr == 0) {
                        return false;
                    }

                    channelTile->extendedInputDimC = channelTile->finalTiles.extendedInputDimC;
                    channelTile->extendedOutputDimC = channelTile->finalTiles.extendedOutputDimC;

                    channelTile->channelStartIndex = socInd * inputTileDims[Dim::C];
                    channelTile->numInputChannels = inputTileDims[Dim::C];

                    planeTile->channelTiles.emplace_back(channelTile);
                }

                _hwTiling->planeTiles.emplace_back(planeTile);
            }
        }
        return true;
    }

    bool hasSoC(const GraphDataTiling& dirTile) const {
        return dirTile.getInputTileDims()[Dim::C] != _convolutionOptions._inputDims[Dim::C];
    }

private:
    const ConvolutionOptions& _convolutionOptions;
    GraphDataTiling& _dirTiling;
    HwConvTilingPtr _hwTiling;
    bool _tileCutPossible;
};
}  // namespace HWTilingNS

}  // namespace vpu
