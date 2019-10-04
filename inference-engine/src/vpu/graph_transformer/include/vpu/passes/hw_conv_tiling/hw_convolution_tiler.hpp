// Copyright (C) 2018-2019 Intel Corporation
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
#include <vpu/hw/tiling.hpp>
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
    ConvolutionOptions(const std::string &stageName, const DimValues &inputDims, const DimValues &outputDims,
                       const DimValues &origOutputDims, const int kernelSizeX, const int kernelSizeY,
                       const int kernelStride, const int paddingLeft, const int paddingRight,
                       const int paddingTop, const int paddingBottom, const bool withPool)
            : _stageName(stageName), _inputDims(inputDims), _outputDims(outputDims),
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

bool operator<(const TilingOption &a, const TilingOption &b);

std::ostream &operator<<(std::ostream &o, const TilingOption &to);

enum class Direction {
    INPUT_TO_OUTPUT = 0, OUTPUT_TO_INPUT = 1
};

// Tensors can be split going either from input to output or vice versa
class GraphDataTiling {
protected:
    const ConvolutionOptions &_co;
    // size of every tile for input tensor in each dimension
    DimValues _inputTileDims;
    // size of every tile for output tensor in each dimension
    DimValues _outputTileDims;
    bool _useCeil = false;
    const enum Direction _direction;

public:
    GraphDataTiling() = delete;
    virtual ~GraphDataTiling() = default;
    GraphDataTiling(const GraphDataTiling &other): _co(other._co), _inputTileDims(other._inputTileDims),
       _outputTileDims(other._outputTileDims), _useCeil(other._useCeil), _direction(other._direction) {
    }

    explicit GraphDataTiling(const ConvolutionOptions &__co, Direction direction) :
            _co(__co), _direction(direction) {}

    const DimValues &getInputTileDims() const { return _inputTileDims; }

    const DimValues &getOutputTileDims() const { return _outputTileDims; }

    DimValues &getInputTileDims() { return _inputTileDims; }

    DimValues &getOutputTileDims() { return _outputTileDims; }

    void resetInputTileDims(const DimValues &dimVals) { _inputTileDims = dimVals; }

    void resetOutputTileDims(const DimValues &dimVals) { _outputTileDims = dimVals; }

    virtual void initTileSizes() = 0;

    virtual void applyTilingOption(const TilingOption &tilingOption) = 0;

    virtual void setInputNOutputTileDimensions(const int tileDimW, const int tileDimH, const int tileDimC) = 0;

    virtual void correctPlaneSize() = 0;

    virtual const DimValues &splitOverTensorDims() = 0;

    virtual void patternMatching() = 0;

    bool useCeil() const {
        return _useCeil;
    }

    Direction getDirection() const {
        return _direction;
    }

    const ConvolutionOptions& co() const { return _co; }
};

class ConvGraphDataTilingFactory final {
public:
    static std::unique_ptr<GraphDataTiling> makeDirTiling(const ConvolutionOptions &co, Direction direction);
    static std::unique_ptr<GraphDataTiling> makeDirTiling(const GraphDataTiling &o);
};

class HWConvolutionTileLayoutCut;

// iterates over all the tiling options and chooses few with minimal cost
class HWConvolutionTilingSearcher {
    const ConvolutionOptions _co;
    const size_t _maxTilingOptions;
    const std::unique_ptr<GraphDataTiling> _dirTiling;
    std::vector<TilingOption> _tilingOptions;

public:
    HWConvolutionTilingSearcher() = delete;
    HWConvolutionTilingSearcher(const HWConvolutionTilingSearcher &other): _co(other._co),
        _maxTilingOptions(other._maxTilingOptions),
        _dirTiling(ConvGraphDataTilingFactory::makeDirTiling(*other._dirTiling)),
        _tilingOptions(other._tilingOptions) {
    }

    HWConvolutionTilingSearcher(const ConvolutionOptions &co,
                                Direction direction,
                                size_t maxTilingOptions) : _co(co),
       _dirTiling(ConvGraphDataTilingFactory::makeDirTiling(_co, direction)),
       _maxTilingOptions(maxTilingOptions) {
        IE_ASSERT(maxTilingOptions > 0);
        _dirTiling->initTileSizes();
        _tilingOptions = selectBetterTiling();
    }

    const std::vector<TilingOption> &tilingOptions() const {
        return _tilingOptions;
    }

    size_t tilingOptionsCount() const {
        return _tilingOptions.size();
    }

    const ConvolutionOptions& co() const { return _co; }

    HWConvolutionTileLayoutCut tileLayoutCut(const TilingOption &option) const;

private:
    std::vector<TilingOption> selectBetterTiling() const;
};

// Search for tiling options and applies them to prepare hw tilings
class HWConvolutionTiler final {
private:
    const ConvolutionOptions _co;
    std::vector<HwConvTilingPtr> _hwTilings;
    bool _tilingPossible;
    const HWConvolutionTilingSearcher _searcher;

public:
    HWConvolutionTiler() = delete;

    HWConvolutionTiler(const HWConvolutionTiler &other): _co(other._co), _hwTilings(other._hwTilings),
        _searcher(other._searcher), _tilingPossible(other._tilingPossible) {
    }

    explicit HWConvolutionTiler(const ConvolutionOptions &co,
                                Direction direction,
                                size_t maxTilingOptions);


    bool isTilingPossible() const {
        return _tilingPossible;
    }

    bool withPool() const {
        return _co._withPool;
    }

    const std::vector<HwConvTilingPtr> &getHwTilings() const {
        return _hwTilings;
    }

private:
    bool tileForHW();
};

SmallVector<HwPlaneTileInfo> calcHeightTiles(const ConvolutionOptions &_co,
                                             const DimValues &outputTileDims, bool useCeil);
SmallVector<HwPlaneTileInfo> calcWidthTiles(const ConvolutionOptions &_co,
                                            const DimValues &outputTileDims, bool useCeil);

// Based on chosen { inputTileDims, outputTileDims } constructs plane's tiling structure;
// (same for both input and output, contains only number of tiles in each dimension)
class HWConvolutionTileLayoutCut {
private:
    const ConvolutionOptions &_co;
    GraphDataTiling &_dirTiling;
    HwConvTilingPtr _hwTiling;
    bool _tileCutPossible;

public:
    HWConvolutionTileLayoutCut() = delete;
    HWConvolutionTileLayoutCut(const HWConvolutionTileLayoutCut &other): _co(other._co), _dirTiling(other._dirTiling),
        _hwTiling(other._hwTiling), _tileCutPossible(other._tileCutPossible) {
    }

    HWConvolutionTileLayoutCut(HWConvolutionTileLayoutCut &&other): _co(other._co), _dirTiling(other._dirTiling) {
        _hwTiling = std::move(other._hwTiling);
        _tileCutPossible = other.tileCutPossible();
    }
    HWConvolutionTileLayoutCut(GraphDataTiling &dirTiling, const TilingOption &tilingOption) :
            _dirTiling(dirTiling),
            _co(dirTiling.co()), _hwTiling(std::make_shared<HwConvTiling>()) {
        dirTiling.applyTilingOption(tilingOption);

        dirTiling.patternMatching();

        // Merged Pooling and SoC can't be used together.
        if (_co._withPool) {
            IE_ASSERT(!hasSoC(dirTiling));
        }

        _tileCutPossible = createTiles(calcHeightTiles(_co, dirTiling.getOutputTileDims(), dirTiling.useCeil()),
                                      calcWidthTiles(_co, dirTiling.getOutputTileDims(), dirTiling.useCeil()),
                                      dirTiling.getInputTileDims(), dirTiling.getOutputTileDims());
    }

    bool tileCutPossible() const { return _tileCutPossible; }

    HwConvTilingPtr hwTiling() const {
        IE_ASSERT(_tileCutPossible);
        return _hwTiling;
    }

private:
    bool createTiles(const SmallVector<HwPlaneTileInfo> &heightTiles,
                     const SmallVector<HwPlaneTileInfo> &widthTiles,
                     const DimValues &inputTileDims, const DimValues &outputTileDims) const {
        IE_ASSERT(!heightTiles.empty());
        IE_ASSERT(!widthTiles.empty());

        _hwTiling->sohTiles = heightTiles.size();
        _hwTiling->sowTiles = widthTiles.size();
        _hwTiling->socTiles = divUp(_co._inputDims[Dim::C], inputTileDims[Dim::C]);

        for (int sohInd = 0; sohInd < _hwTiling->sohTiles; ++sohInd) {
            const auto &heightTileInfo = heightTiles[sohInd];

            for (int sowInd = 0; sowInd < _hwTiling->sowTiles; ++sowInd) {
                const auto &widthTileInfo = widthTiles[sowInd];

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
                            widthTileInfo.inputWithJunk, heightTileInfo.inputWithJunk, inputTileDims[Dim::C],
                            outputTileDims[Dim::C],
                            _co._kernelSizeX, _co._kernelSizeY, _co._kernelStride);

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

    bool hasSoC(const GraphDataTiling &dirTile) const {
        return dirTile.getInputTileDims()[Dim::C] != _co._inputDims[Dim::C];
    }
};
}  // namespace HWTilingNS

}  // namespace vpu
