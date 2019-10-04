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
#include <vpu/passes/hw_conv_tiling/hw_convolution_tiler.hpp>

namespace vpu {

namespace HWTilingNS {

using HWTilingNS::GraphDataTiling;
using HWTilingNS::ConvolutionOptions;
using HWTilingNS::Direction;
using HWTilingNS::TilingOption;

const int CHANS_PER_DESCR = 16;

HwPoolTileInfo splitPooling(int outZ);

class PoolGraphDataTilingFactory final {
public:
    static std::unique_ptr<GraphDataTiling> makeDirTiling(const ConvolutionOptions &co, Direction direction);
    static std::unique_ptr<GraphDataTiling> makeDirTiling(const GraphDataTiling &o);
};

class HWPoolingTileLayoutCut;

// iterates over all the tiling options and chooses few with minimal cost
class HWPoolingTilingSearcher {
    const ConvolutionOptions _co;
    const size_t _maxTilingOptions;
    const std::unique_ptr<GraphDataTiling> _dirTiling;
    std::vector<TilingOption> _tilingOptions;

public:
    HWPoolingTilingSearcher() = delete;
    HWPoolingTilingSearcher(const HWPoolingTilingSearcher &other): _co(other._co),
       _maxTilingOptions(other._maxTilingOptions),
       _dirTiling(PoolGraphDataTilingFactory::makeDirTiling(*other._dirTiling)),
       _tilingOptions(other._tilingOptions) {
    }

    HWPoolingTilingSearcher(const ConvolutionOptions &co,
                                Direction direction,
                                size_t maxTilingOptions) : _co(co),
                           _dirTiling(PoolGraphDataTilingFactory::makeDirTiling(_co, direction)),
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

    const HWPoolingTileLayoutCut tileLayoutCut(const TilingOption &option) const;

private:
    std::vector<TilingOption> selectBetterTiling() const;
};

// Search for tiling options and applies them to prepare hw tilings
class HWPoolingTiler final {
private:
    const ConvolutionOptions _co;
    std::vector<HwPoolTilingPtr> _hwTilings;
    bool _tilingPossible;
    const HWPoolingTilingSearcher _searcher;

public:
    HWPoolingTiler() = delete;

    HWPoolingTiler(const HWPoolingTiler &other): _co(other._co), _hwTilings(other._hwTilings),
                                                         _searcher(other._searcher), _tilingPossible(other._tilingPossible) {
    }

    explicit HWPoolingTiler(const ConvolutionOptions &co,
                                Direction direction,
                                size_t maxTilingOptions);

    bool isTilingPossible() const {
        return _tilingPossible;
    }

    const std::vector<HwPoolTilingPtr> &getHwTilings() const {
        return _hwTilings;
    }

private:
    bool tileForHW();
};

SmallVector<HwPlaneTileInfo> calcHeightTilesP(const ConvolutionOptions &_co,
                                              const DimValues &outputTileDims, bool useCeil);
SmallVector<HwPlaneTileInfo> calcWidthTilesP(const ConvolutionOptions &_co,
                                             const DimValues &outputTileDims, bool useCeil);

// Based on chosen { inputTileDims, outputTileDims } constructs plane's tiling structure;
// (same for both input and output, contains only number of tiles in each dimension)
class HWPoolingTileLayoutCut {
private:
    const ConvolutionOptions &_co;
    GraphDataTiling &_dirTiling;
    HwPoolTilingPtr _hwTiling;
    bool _tileCutPossible;

public:
    HWPoolingTileLayoutCut() = delete;
    HWPoolingTileLayoutCut(const HWPoolingTileLayoutCut &other): _co(other._co), _dirTiling(other._dirTiling),
                                               _hwTiling(other._hwTiling), _tileCutPossible(other._tileCutPossible) {
    }

    HWPoolingTileLayoutCut(HWPoolingTileLayoutCut &&other): _co(other._co), _dirTiling(other._dirTiling) {
        _hwTiling = std::move(other._hwTiling);
        _tileCutPossible = other.tileCutPossible();
    }
    HWPoolingTileLayoutCut(GraphDataTiling &dirTiling, const TilingOption &tilingOption) :
            _dirTiling(dirTiling),
            _co(dirTiling.co()), _hwTiling(std::make_shared<HwPoolTiling>()) {
        dirTiling.applyTilingOption(tilingOption);

        _tileCutPossible = createTiles(calcHeightTilesP(_co, dirTiling.getOutputTileDims(), dirTiling.useCeil()),
                                       calcWidthTilesP(_co, dirTiling.getOutputTileDims(), dirTiling.useCeil()),
                                       dirTiling.getInputTileDims(), dirTiling.getOutputTileDims());
    }

    bool tileCutPossible() const { return _tileCutPossible; }

    HwPoolTilingPtr hwTiling() const {
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
        _hwTiling->socTiles = divUp(_co._inputDims.get(Dim::N, 1), inputTileDims[Dim::N]);

        for (int sohInd = 0; sohInd < _hwTiling->sohTiles; ++sohInd) {
            const auto& heightTileInfo = heightTiles[sohInd];

            for (int sowInd = 0; sowInd < _hwTiling->sowTiles; ++sowInd) {
                const auto& widthTileInfo = widthTiles[sowInd];

                auto planeTile = std::make_shared<HwPoolPlaneTile>();
                planeTile->parent = _hwTiling;

                planeTile->sohInd = sohInd;
                planeTile->sowInd = sowInd;

                planeTile->heightInfo = heightTileInfo;
                planeTile->widthInfo = widthTileInfo;

                for (int socInd = 0; socInd < _hwTiling->socTiles; ++socInd) {
                    auto channelTile = std::make_shared<HwPoolChannelTile>();
                    channelTile->parent = planeTile;

                    channelTile->socInd = socInd;

                    channelTile->finalTiles = splitPooling(inputTileDims[Dim::C] * inputTileDims[Dim::N]);

                    if (channelTile->finalTiles.numDescr == 0) {
                        return false;
                    }

                    channelTile->channelStartIndex = socInd * inputTileDims[Dim::N];
                    channelTile->numInputChannels = inputTileDims[Dim::N];

                    planeTile->channelTiles.emplace_back(channelTile);
                }

                _hwTiling->planeTiles.emplace_back(planeTile);
            }
        }

        return true;
    }
};

}  // namespace HWTilingNS

}  // namespace vpu
