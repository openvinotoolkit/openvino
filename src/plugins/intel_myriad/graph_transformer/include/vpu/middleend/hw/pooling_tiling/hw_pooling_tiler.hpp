// Copyright (C) 2018-2022 Intel Corporation
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
#include <vpu/compile_env.hpp>
#include <vpu/utils/heap.hpp>
#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/conv_tiling/hw_convolution_tiler.hpp>

namespace vpu {

namespace HWTilingNS {

using HWTilingNS::GraphDataTiling;
using HWTilingNS::ConvolutionOptions;
using HWTilingNS::Direction;
using HWTilingNS::TilingOption;

constexpr int CHANNELS_PER_DESCRIPTOR = 16;

HwPoolTileInfo splitPooling(int outZ);

class PoolGraphDataTilingFactory final {
public:
    static std::unique_ptr<GraphDataTiling> makeDirTiling(const ConvolutionOptions& convolutionOptions,
                                                          const Direction& direction);
    static std::unique_ptr<GraphDataTiling> makeDirTiling(const GraphDataTiling& graphDataTiling);
};

class HWPoolingTileLayoutCut;

// iterates over all the tiling options and chooses few with minimal cost
class HWPoolingTilingSearcher {
public:
    HWPoolingTilingSearcher() = delete;

    HWPoolingTilingSearcher(const HWPoolingTilingSearcher& other):
        _convolutionOptions(other._convolutionOptions),
        _maxTilingOptions(other._maxTilingOptions),
        _dirTiling(PoolGraphDataTilingFactory::makeDirTiling(*other._dirTiling)),
        _tilingOptions(other._tilingOptions) {}
    HWPoolingTilingSearcher(const ConvolutionOptions& convolutionOptions, const Direction& direction,
                            std::size_t maxTilingOptions) :
        _convolutionOptions(std::move(convolutionOptions)),
        _dirTiling(PoolGraphDataTilingFactory::makeDirTiling(_convolutionOptions, direction)),
        _maxTilingOptions(maxTilingOptions) {
        IE_ASSERT(maxTilingOptions > 0);
        _dirTiling->initTileSizes();
        _tilingOptions = selectBetterTiling();
    }

    const std::vector<TilingOption>& tilingOptions() const {
        return _tilingOptions;
    }

    const ConvolutionOptions& convolutionOptions() const { return _convolutionOptions; }

    HWPoolingTileLayoutCut tileLayoutCut(const TilingOption& option) const;

private:
    std::vector<TilingOption> selectBetterTiling() const;

    const ConvolutionOptions _convolutionOptions;
    const std::size_t _maxTilingOptions;
    const std::unique_ptr<GraphDataTiling> _dirTiling;
    std::vector<TilingOption> _tilingOptions;
};

// Search for tiling options and applies them to prepare hw tilings
class HWPoolingTiler final {
public:
    HWPoolingTiler() = delete;

    HWPoolingTiler(const HWPoolingTiler&) = default;
    HWPoolingTiler(const ConvolutionOptions& convolutionOptions, const Direction& direction, std::size_t maxTilingOptions);

    bool isTilingPossible() const {
        return _tilingPossible;
    }

    const std::vector<HwPoolTilingPtr>& getHwTilings() const {
        return _hwTilings;
    }

private:
    bool tileForHW();

    const ConvolutionOptions _convolutionOptions;
    std::vector<HwPoolTilingPtr> _hwTilings;
    bool _tilingPossible;
    const HWPoolingTilingSearcher _searcher;
};

SmallVector<HwPlaneTileInfo> calcHeightTilesP(const ConvolutionOptions& convolutionOptions,
                                              const DimValues& outputTileDims, bool useCeil);
SmallVector<HwPlaneTileInfo> calcWidthTilesP(const ConvolutionOptions& convolutionOptions,
                                             const DimValues& outputTileDims, bool useCeil);

// Based on chosen { inputTileDims, outputTileDims } constructs plane's tiling structure;
// (same for both input and output, contains only number of tiles in each dimension)
class HWPoolingTileLayoutCut {
public:
    HWPoolingTileLayoutCut() = delete;

    HWPoolingTileLayoutCut(const HWPoolingTileLayoutCut&) = default;
    HWPoolingTileLayoutCut(HWPoolingTileLayoutCut&& other) noexcept :
        _convolutionOptions(other._convolutionOptions), _dirTiling(other._dirTiling) {
        _hwTiling = std::move(other._hwTiling);
        _tileCutPossible = other.tileCutPossible();
    }
    HWPoolingTileLayoutCut(GraphDataTiling& dirTiling, const TilingOption& tilingOption) :
        _dirTiling(dirTiling),
        _convolutionOptions(dirTiling.convolutionOptions()),
        _hwTiling(std::make_shared<HwPoolTiling>()) {
        dirTiling.applyTilingOption(tilingOption);

        const auto& heightTiles = calcHeightTilesP(
            _convolutionOptions,
            dirTiling.getOutputTileDims(),
            dirTiling.useCeil());

        const auto& widthTiles = calcWidthTilesP(
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

    HwPoolTilingPtr hwTiling() const {
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
        _hwTiling->socTiles = divUp(_convolutionOptions._inputDims.get(Dim::N, 1), inputTileDims[Dim::N]);

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

private:
    const ConvolutionOptions& _convolutionOptions;
    GraphDataTiling& _dirTiling;
    HwPoolTilingPtr _hwTiling;
    bool _tileCutPossible;
};

}  // namespace HWTilingNS

}  // namespace vpu
