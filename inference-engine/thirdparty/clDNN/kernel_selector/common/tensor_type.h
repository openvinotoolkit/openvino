// Copyright (c) 2016-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#include "common_types.h"
#include "common_tools.h"
#include <vector>
#include <assert.h>
#include <numeric>
#include <cstddef>
#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <stdexcept>
#include <map>

namespace kernel_selector {
#define KERNEL_SELECTOR_TENSOR_DIM_MAX 9

namespace Tensor {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataLayout
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum DataLayout {
    f = 0,                  // 1D
    bf,                     // 1D+batch
    fb,                     // 1D+batch
    bfyx,                   // 3D+batch
    yxfb,                   // 3D+batch
    byxf,                   // 3D+batch
    fyxb,                   // 3D+batch
    bfxy,                   // 3D+batch
    b_fs_yx_fsv16,          // 3D+batch
    b_fs_zyx_fsv16,         // batch, feature, 3D spatial. Blocks of 16 input channels
    b_fs_yx_fsv32,          // 3D+batch
    b_fs_zyx_fsv32,         // 4D+batch
    bs_fs_yx_bsv16_fsv16,   // batch, feature, 2D spatial. Blocks of 16 batch and channels
    bs_fs_zyx_bsv16_fsv16,  // batch, feature, 3D spatial. Blocks of 16 batch and channels
    bs_f_bsv8__af8,         // for optimized FC
    bs_f_bsv16__af8,        // for optimized FC
    bf8_xy16,               // for optimized conv1x1
    winograd_2x3_s1_data,   // winograd convolution input, F(2,3) -- filter 3x3 with stride 1
    byxf_af32,              // for MMAD convolution
    byx8_f4,                // for MMAD convolution
    fs_bs_yx_bsv4_fsv32,    // for batched MMAD
    b_fs_yx_fsv4,           // reordering format for swizzled input for convolution using IMAD
    bfzyx,                  // batch+feature+3D spatial
    fs_b_yx_fsv32,          // for FP16 kernels, 32 features to avoid partial writes
    b_fs_yx_32fp,           // bfyx with blocks of 16 packed binary input channels
    bfwzyx,                 // batch, feature, 4D spatial
    nv12,                   // media nv12 layout
    image_2d_rgba,          // image2d RGBA
    DataLayoutCount         // NUMBER OF ELEMENTS IN ENUM
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WeightsLayout
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum WeightsLayout {
    oi = 0,
    io,
    oiyx,
    oyxi,
    iyxo,
    yxio,
    os_iyx_osv16,
    os_iyx_osv32,
    os_iyx_osv32__ai32,
    os_iyx_osv64,
    os_is_zyx_isv16_osv16,
    is_os_zyx_osv16_isv16,
    is_os_yx_osv16_isv16,
    os_is_zyx_isv8_osv16_isv2,
    os_is_yx_isv8_osv16_isv2,
    os_is_yx_isv16_osv16,
    os_zyxi_osv16,
    os_iyx_osv16_rotate_180,
    os_i_osv8__ai8,  // TODO can we drop the alignment form layout name?
    os_i_osv16__ai8,
    os_i_osv16,
    os_is_yx_osv16_isv16,           // wieghts for int8 blocked conv
    os_is_zyx_osv32_isv16,
    os_is_zyx_osv64_isv16,
    i_yxs_os_yxsv2_osv16,
    iy_xs_os_xsv2_osv16__ao32,
    iy_xs_os_xsv2_osv8__ao32,
    image_2d_weights_c4_fyx_b,      // image type fyx_b
    image_2d_weights_c1_b_fyx,      // image type 2d b_fyx single channel
    winograd_2x3_s1_weights,        // winograd convolution weights, F(2, 3) --filter 3x3 with stride 1
    winograd_2x3_s1_fused_weights,  // winograd convolution weights for fused kernel, F(2, 3) --filter 3x3 with stride 1
    winograd_6x3_s1_fused_weights,  // winograd convolution weights for fused kernel, F(6, 3) --filter 3x3 with stride 1
    image_2d_weights_winograd_6x3_s1_fbxyb,  // image 2d winograd convolution weights for fused kernel, F(2, 3) --filter
                                             // 3x3 with stride 1
    image_2d_weights_winograd_6x3_s1_xfbyb,  // image 2d winograd convolution weights for fused kernel, F(2, 3) --filter
                                             // 3x3 with stride 1
    dlstm_dir_io,                            // dlstm weights layout direction, input_size, 4* hiden_size
    os_is_yx_isa8_osv8_isv4,                 // for MMAD convolution
    os_is_zyx_isa8_osv8_isv4,                // for MMAD convolution
    os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,  // for MMAD convolution swizzled from ofm 0..7 to 0,4,8,12,16,20,24,28,
                                                 // 1,5...
    os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4,  // for MMAD convolution swizzled from ofm 0..7 to 0,4,8,12,16,20,24,28,
                                                  // 1,5...
    os_is_yx_isa8_osv8_isv4_swizzled_by_4,   // for MMAD convolution swizzled from ofm 0..7 to 0,4,8,12,16,20,24,28,
                                             // 1,5...
    is_o_yx_isv32,                           // for MMAD 1x1 convolutions
    is_o32_yx_isv32_swizzled_by_4,  // for MMAD 1x1 convolutions swizzled from ofm 0..7 to 0,4,8,12,16,20,24,28, 1,5...
    os_is_y_x8_osv8_isv4,           // for MMAD convolutions
    os_is_y_x8_osv8_isv4_swizzled_by_4,  // for MMAD 1x1 convolutions swizzled from ofm 0..7 to 0,4,8,12,16,20,24,28,
                                         // 1,5...
    os_is_yx_osv16_isv4,                 // swizzled weights for convolution using IMAD
    os_is_yx_osv32_isv4_swizzled_by_2,   //  weights for bfyx -> b_fs_yx_fsv32 convolution using IMAD with swizzeled ofm (0, 2, 4..), (1, 3, 5...)
    os_is_yx_osv32_isv4,                 //  weights for bfyx -> b_fs_yx_fsv{32,16} convolution using IMAD
    oizyx,
    os_is_yx_osv32_isv32p,  // 2 blocks: 32 packed binary in channels and 32 output channels
    os_is_osv32_isv32_swizzled_by_4,     // for weights for 1x1 IMAD convolution
    os_i_yxs_osv4_yxsv4,                 // for weights for depthwise IMAD convolution
    goiyx,
    goizyx,
    gyxio,
    g_os_iyx_osv16,
    g_os_iyx_osv32,
    gs_oiyx_gsv16,
    gs_oizyx_gsv16,
    gs_oiyx_gsv32,
    g_os_iyx_osv16_rotate_180,
    gi_yxs_os_yxsv2_osv16,
    g_is_os_zyx_osv16_isv16,
    g_is_os_yx_osv16_isv16,
    g_os_is_zyx_isv8_osv16_isv2,
    g_os_is_yx_isv8_osv16_isv2,
    g_os_is_zyx_isv16_osv16,
    giy_xs_os_xsv2_osv16__ao32,
    giy_xs_os_xsv2_osv8__ao32,
    g_os_is_yx_isv16_osv16,
    gs_oi_yxs_gsv4_yxsv4,                // grouped weights for depthwise IMAD convolution (b_fs_yx_fsv4 format)
    gs_oi_yxs_gsv16_yxsv4,               // grouped weights for depthwise IMAD convolution (b_fs_yx_fsv16 format)
    gs_oi_yxs_gsv32_yxsv4,               // grouped weights for depthwise IMAD convolution (b_fs_yx_fsv32 format)

    g_os_is_yx_osv16_isv4,

    g_os_zyx_is_osv16_isv4,
    g_os_zyx_is_osv16_isv16,
    g_os_zyx_is_osv16_isv32,
    g_os_zyx_is_osv32_isv4,
    g_os_zyx_is_osv32_isv16,
    g_os_zyx_is_osv32_isv32,

    WeightsLayoutCount                   // NUMBER OF ELEMENTS IN ENUM
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pad
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Pad {
    size_t before;
    size_t after;

    size_t Total() const { return before + after; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dim
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Dim {
    // PitchesDesc sorted in order from often to rare, where pair is a Pitch and
    // a corresponding Block size for current dimension
    using PitchesDesc = std::vector<std::pair<size_t, size_t>>;

    size_t v;
    PitchesDesc pitches;  
    Pad pad;

    Dim() = default;
    Dim(size_t v, PitchesDesc pitches, Pad pad) : v(v), pitches(pitches), pad(pad) {}
    Dim(size_t v, size_t pitch, Pad pad) : v(v), pad(pad) {
        SetLinearPitch(pitch);
    }

    size_t Pitch() const { return pitches.back().first; }
    size_t BlockSize() const { return pitches.back().second; }
    size_t LogicalDimPadded() const { return v + pad.Total(); }
    void SetLinearPitch(size_t pitch, size_t block_size = 1) {
        std::pair<size_t, size_t> desc = {pitch, block_size};
        pitches = {desc};
    }
};

using NDims = std::vector<Dim>;
using BlockSizesDesc = std::vector<std::pair<int, size_t>>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extract code
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class DataChannelName { X = 0, Y = 1, Z = 2, W = 3, FEATURE = 4, BATCH = 5, COUNT = 6 };

enum class WeightsChannelName { X = 0, Y = 1, Z = 2, IFM = 3, OFM = 4, LX = 5, LY = 6, G = 7, COUNT = 8 };

inline bool SimpleLayout(WeightsLayout l) {
    switch (l) {
        case WeightsLayout::oi:
        case WeightsLayout::io:
        case WeightsLayout::oiyx:
        case WeightsLayout::oyxi:
        case WeightsLayout::iyxo:
        case WeightsLayout::yxio:
        case WeightsLayout::oizyx:
        case WeightsLayout::dlstm_dir_io:
            return true;
        default:
            return false;
    }
}

inline bool SimpleLayout(DataLayout l) {
    switch (l) {
        case DataLayout::bf:
        case DataLayout::fb:
        case DataLayout::bfyx:
        case DataLayout::yxfb:
        case DataLayout::byxf:
        case DataLayout::fyxb:
        case DataLayout::bfzyx:
        case DataLayout::bfwzyx:
            return true;
        default:
            return false;
    }
}

inline bool GroupedLayout(WeightsLayout l);

inline bool GroupedLayout(DataLayout) {
    return false;
}

inline bool IsImageType(WeightsLayout l) {
    switch (l) {
        case WeightsLayout::image_2d_weights_c4_fyx_b:
        case WeightsLayout::image_2d_weights_c1_b_fyx:
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb:
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb:
            return true;
        default:
            return false;
    }
}

inline bool IsDynamicLSTMType(WeightsLayout l) {
    switch (l) {
    case WeightsLayout::dlstm_dir_io:
        return true;
    default:
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tensor Exaplnation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// resource     - 80x80
//      totalSize   - 6400
//      x pitch     - 1
//      y pitch     - 80
//
// view         - 60x60
// viewOffset   - (20,20) => 20*80+20 = 1620
//
// padding (contains "paddedVal"):
//      before  - x=20, y=20
//      after   - x=20, y=20.
//
// logical data - 40x40 (contains the actual data).
//
// firstElementOffset:
//      (viewOffset_x + padBefore_x) + (viewOffset_y + padBefore_y)*y_pitch =
//      viewOffset + padBefore_x + padBefore_y*y_pitch =
//      1620 + 20 + 20*80 = 3240
//
//
//                                      whole resource (80x80)
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                                               +
// +                                                                                               +
// +                                view inside resource (60x60)                                   +
// +       +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       +
// +       + start of padded part(20,20) = viewOffset                                      +       +
// +       +                                                                               +       +
// +       +                             logical data (40x40)                              +       +
// +       +       +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       +       +
// +       +       + first element (40,40)                                         +       +       +
// +       +       +                                                               +       +       +
// +       +       +                                                               +       +       +
// +       +       +                                                               +       +       +
// +       +       +                                                               +       +       +
// +       +       +                                                               +       +       +
// +       +       +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       +       +
// +       +                                                                               +       +
// +       +                                                                               +       +
// +       +                                                                               +       +
// +       +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       +
// +                                                                                               +
// +                                                                                               +
// +                                                                                               +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TensorBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TensorBase {
protected:
    NDims dims;
    NDims blockedDims;
    size_t viewOffset = 0;  // in elements
    size_t firstElementOffset = 0;
    size_t totalSize = 0;  // in elements
    float paddedVal = 0.f;

public:
    TensorBase() = default;
    TensorBase(const TensorBase&) = default;
    TensorBase& operator=(const TensorBase&) = default;

    TensorBase(const NDims& nd, const BlockSizesDesc& blockSizes, size_t viewOf, size_t sz, float pv)
        : dims(nd),
          viewOffset(viewOf),
          totalSize(sz),
          paddedVal(pv) {
        // This pass fills `blockedDims` and calculates correct pitches for `dims` if `blockSizes` specified for Layout
        if (!blockSizes.empty()) {
            size_t blocksNum = blockSizes.size();
            std::vector<size_t> dimsBlocked(dims.size() + blocksNum);
            std::vector<size_t> pitchesBlocked(dims.size() + blocksNum);

            // Add to dimsBlocked the most frequently changed values first (block sizes) and tensor values after
            for (size_t i = 0; i < dimsBlocked.size(); i++) {
                dimsBlocked[i] = (i < blocksNum) ? blockSizes[i].second : dims[i - blocksNum].LogicalDimPadded();
            }

            // Recalculate linear channel's value according to specifed block sizes
            for (auto& bs : blockSizes) {
                const size_t blockChannelIdx = bs.first;
                const size_t blockedDim = bs.second;
                const size_t simpleDim = dimsBlocked[blocksNum + blockChannelIdx];
                dimsBlocked[blocksNum + blockChannelIdx] = CeilDiv(simpleDim, blockedDim);
            }

            // Calculate correct pitches
            pitchesBlocked[0] = 1;
            for (size_t i = 1; i < pitchesBlocked.size(); i++) {
                pitchesBlocked[i] = pitchesBlocked[i - 1] * dimsBlocked[i - 1];
            }

            // Add blocked pitches to original `dims`
            for (size_t i = 0; i < blocksNum; i++) {
                const size_t index = blocksNum - i - 1;
                const std::pair<size_t, size_t> pitch = {pitchesBlocked[index], blockSizes[index].second};
                dims[blockSizes[index].first].pitches.insert(dims[blockSizes[index].first].pitches.begin(), pitch);
            }

            auto blockSizeMul = [](size_t val, const std::pair<size_t, size_t>& d) {
                return val * d.second;
            };

            // Update linear pitches in original `dims`
            for (size_t i = 0; i < dims.size(); i++) {
                // Multiply all block sizes across channel (except last value that related to linear size)
                const size_t totalBlockSize = std::accumulate(dims[i].pitches.begin(), std::prev(dims[i].pitches.end()), (size_t)1, blockSizeMul);
                dims[i].pitches.back() = {pitchesBlocked[blocksNum + i], totalBlockSize};
            }

            // Add all dims into `blockedDims` (in order from often to rare)
            for (size_t i = 0; i < dimsBlocked.size(); i++) {
                const std::pair<size_t, size_t> pitch = {pitchesBlocked[i], 1};
                const Pad pad = (i >= blocksNum) ? dims[i - blocksNum].pad : Pad{0,0};
                blockedDims.push_back({dimsBlocked[i], {pitch}, pad});
            }
        }

        auto calcOffset = [](size_t val, const Dim& d) {
            size_t offset = 0;
            for (auto p = d.pitches.begin(); p != std::prev(d.pitches.end()); p++) {
                offset += (d.pad.before % p->second) * p->first;  // Calculate offset in block first
            }
            offset += (d.pad.before / d.pitches.back().second) * d.pitches.back().first;
            return val + offset;
        };

        this->firstElementOffset = std::accumulate(nd.cbegin(), nd.cend(), viewOf, calcOffset);

        if (totalSize == 0) {
            for (const auto& d : dims) {
                for (const auto& p: d.pitches) {
                    totalSize = std::max(totalSize, p.first * (d.LogicalDimPadded()));
                }
            }

            totalSize += viewOffset;
        }

        // If correct block sizes have not been set then use 1 as minimal pitch
        size_t minimalPitch = (blockSizes.empty()) ? 1 : dims[0].Pitch();

        for (const auto& d : dims) {
            if (d.Pitch() < minimalPitch) {
                throw std::runtime_error("Tensor pitches didn't set correctly");
            }

            minimalPitch *= CeilDiv(d.LogicalDimPadded(), d.BlockSize());
        }

        if (totalSize < (minimalPitch + viewOffset)) {
            throw std::runtime_error("Tensor total Size didn't set correctly");
        }
    }

    float GetPaddedVal() const { return paddedVal; }
    size_t GetFirstElementOffset() const { return firstElementOffset; }
    size_t GetViewOffset() const { return viewOffset; }
    const NDims& GetDims() const { return dims; }
    const NDims& GetBlockedDims() const { return blockedDims; }

    virtual uint32_t ElementSize() const = 0;

    // Size of the actual data (without padded part)
    size_t LogicalSize() const {
        return std::accumulate(dims.cbegin(), dims.cend(), (size_t)1, [](size_t val, const Dim& d) {
            return val * d.v;
        });
    }

    // Dimensions of the actual data (without padded part)
    std::vector<size_t> LogicalDims() const {
        std::vector<size_t> res(dims.size());
        std::transform(dims.begin(), dims.end(), res.begin(), [](const Dim& d) { return d.v; });
        return res;
    }

    // Whole buffer size (in elements)
    size_t PhysicalSize() const { return totalSize; }

    // Whole buffer size (in bytes)
    size_t PhysicalSizeInBytes() const { return totalSize * ElementSize(); }

    // if padded/view exists between logical dimensions.
    // in other words, if we can consider the data as a 1Dim resource.
    bool PitchesDifferFromLogicalDims() const {
        bool differ = false;

        size_t calc_pitch = (blockedDims.empty()) ? 1 : dims[0].Pitch();
        for (const auto& d : dims) {
            differ |= (d.Pitch() != calc_pitch);
            calc_pitch *= CeilDiv(d.v, d.BlockSize());
        }

        return differ;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TensorBaseT
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DType, typename Layout>
struct TensorBaseT : public TensorBase {
protected:
    DType dtype;
    Layout layout;

    template <typename ArrayT, typename ChannelName>
    static inline int ChannelIndex(const ArrayT& channelArr, Layout l, ChannelName channelName) {
        size_t channel = static_cast<size_t>(channelName);

        for (auto& entry : channelArr) {
            if (entry.first == l)
                return entry.second.channelsOrder[channel];
        }

        return -1;
    }

    template <typename ArrayT, typename ChannelName>
    static inline Dim Extract(const ArrayT& channelArr, Layout l, ChannelName channelName, const NDims& dims) {
        const int i = ChannelIndex(channelArr, l, channelName);
        return ((i < 0) || (i >= static_cast<int>(dims.size()))) ? Dim{1, {{1, 1}}, {0, 0}} : dims[i];
    }

    template <typename ArrayT>
    static inline uint32_t ChannelsCount(const ArrayT& channelArr, Layout l) {
        const auto& entry = channelArr.find(l);

        if (entry == channelArr.end())
            throw std::invalid_argument("Failed to get channels count for layout " + std::to_string(static_cast<uint32_t>(l)));

        return std::accumulate(entry->second.channelsOrder.begin(), entry->second.channelsOrder.end(), 0U, [](uint32_t count, int v) {
            return count + ((v != -1) ? 1 : 0);
        });
    }

    template <typename ArrayT>
    static inline uint32_t BlockedChannelsCount(const ArrayT& channelArr, Layout l) {
        const auto& entry = channelArr.find(l);

        if (entry == channelArr.end())
            throw std::invalid_argument("Failed to get channels count for layout " + std::to_string(static_cast<uint32_t>(l)));

        return static_cast<uint32_t>(entry->second.blockedChannels.size());
    }

    template <typename ArrayT, typename ChannelName>
    static inline BlockSizesDesc GetBlockedChannels(const ArrayT& channelArr, Layout l) {
        const auto& entry = channelArr.find(l);

        if (entry == channelArr.end())
            throw std::invalid_argument("Failed to get blocked channels for layout " +  std::to_string(static_cast<uint32_t>(l)));

        const auto& blockedChannels = entry->second.blockedChannels;
        BlockSizesDesc res(blockedChannels.size());

        std::transform(blockedChannels.begin(), blockedChannels.end(), res.begin(), [&](const std::pair<ChannelName, size_t>& p) {
            return std::make_pair(ChannelIndex(channelArr, l, p.first), p.second);
        });

        return res;
    }

    template <typename ArrayT, typename ChannelName>
    static inline std::vector<int> GetSortedChannelsIndicesT(const ArrayT& channelArr, Layout l) {
        std::vector<int> res;
        for (size_t i = 0; i < static_cast<size_t>(ChannelName::COUNT); i++) {
            int idx = ChannelIndex(channelArr, l, static_cast<ChannelName>(i));
            if (idx != -1)
                res.push_back(idx);
        }
        return res;
    }

public:
    TensorBaseT() = default;
    TensorBaseT(const TensorBaseT&) = default;
    TensorBaseT& operator=(const TensorBaseT&) = default;

    TensorBaseT(const NDims& nd, const BlockSizesDesc& ndBlocked, DType dt, Layout l, size_t of = 0, size_t sz = 0, float pv = 0.f)
        : TensorBase(nd, ndBlocked, of, sz, pv), dtype(dt), layout(l) {}

    DType GetDType() const { return dtype; }
    Layout GetLayout() const { return layout; }
    uint32_t ElementSize() const override { return BytesPerElement(dtype); }
    size_t Dimentions() const { return dims.size(); }
    bool SimpleLayout() const { return Tensor::SimpleLayout(layout); }
    bool GroupedLayout() const { return Tensor::GroupedLayout(layout); }

    bool operator==(const TensorBaseT& t) const {
        bool same = dtype == t.dtype && layout == t.layout && paddedVal == t.paddedVal && viewOffset == t.viewOffset &&
                    dims.size() == t.dims.size();
        if (same) {
            for (size_t i = 0; i < dims.size(); i++) {
                same &= dims[i].v == t.dims[i].v && dims[i].pad.before == t.dims[i].pad.before &&
                        dims[i].pad.after == t.dims[i].pad.after && dims[i].pitches.size() == t.dims[i].pitches.size();
                if (same) {
                    for (size_t p = 0; p < dims[i].pitches.size(); p++) {
                        same &= dims[i].pitches[p].first == t.dims[i].pitches[p].first &&
                                dims[i].pitches[p].second == t.dims[i].pitches[p].second;
                    }
                }
            }
        }

        return same;
    }

    bool SameDims(const TensorBaseT& t) const {
        bool same = dtype == t.dtype && layout == t.layout && dims.size() == t.dims.size();
        if (same) {
            for (size_t i = 0; i < dims.size(); i++) {
                same &= dims[i].v == t.dims[i].v;
            }
        }

        return same;
    }

    bool SameDimsSizes(const TensorBaseT& t) const {
        bool same = dims.size() == t.dims.size();
        if (same) {
            for (size_t i = 0; i < dims.size(); i++) {
                same &= dims[i].v == t.dims[i].v;
            }
        }

        return same;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataTensor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct DataTensor : public TensorBaseT<Datatype, DataLayout> {
    DataTensor() = default;
    DataTensor(const DataTensor&) = default;
    DataTensor& operator=(const DataTensor&) = default;

    DataTensor(const NDims& nd, Datatype dt, DataLayout l, size_t of = 0, size_t sz = 0, float pv = 0.f)
        : TensorBaseT(nd, GetBlocked(l), dt, l, of, sz, pv) {}

    DataTensor(const std::vector<size_t>& d, Datatype dt, DataLayout l)
        : TensorBaseT<Datatype, DataLayout>(GetSimpleDims(d, l), GetBlocked(l), dt, l) {}

    Dim X() const { return Extract(layout, DataChannelName::X, dims); }
    Dim Y() const { return Extract(layout, DataChannelName::Y, dims); }
    Dim Z() const { return Extract(layout, DataChannelName::Z, dims); }
    Dim W() const { return Extract(layout, DataChannelName::W, dims); }
    Dim Feature() const { return Extract(layout, DataChannelName::FEATURE, dims); }
    Dim Batch() const { return Extract(layout, DataChannelName::BATCH, dims); }

    DataTensor TransformIgnorePadding(DataLayout l) const;
    DataTensor FlattenFeatureAndSpatials() const;
    DataTensor FlattenEverything() const;

    static inline Dim Extract(DataLayout l, DataChannelName channel, const NDims& d) {
        return TensorBaseT::Extract(dataChannelArray, l, channel, d);
    }

    static inline int Channelndex(DataLayout l, DataChannelName channel) {
        return TensorBaseT::ChannelIndex(dataChannelArray, l, channel);
    }

    static inline uint32_t ChannelsCount(DataLayout l) { return TensorBaseT::ChannelsCount(dataChannelArray, l); }

    static inline uint32_t BlockedChannelsCount(DataLayout l) { return TensorBaseT::BlockedChannelsCount(dataChannelArray, l); }

    static inline std::vector<int> GetSortedChannelsIndices(DataLayout l) {
        return TensorBaseT::GetSortedChannelsIndicesT<DataChannelArray, DataChannelName>(dataChannelArray, l);
    }
private:
    struct DataChannelDesc {
        std::array<int, static_cast<size_t>(DataChannelName::COUNT)> channelsOrder;
        std::vector<std::pair<DataChannelName, size_t>> blockedChannels;
    };

    using DataChannelArray = std::map<DataLayout, DataChannelDesc>;
    static DataChannelArray dataChannelArray;
    static NDims GetSimpleDims(const std::vector<size_t>& d, DataLayout l);
    static inline BlockSizesDesc GetBlocked(DataLayout l) {
        return TensorBaseT::GetBlockedChannels<DataChannelArray, DataChannelName>(dataChannelArray, l);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WeightsTensor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct WeightsTensor : TensorBaseT<WeightsType, WeightsLayout> {
    WeightsTensor() = default;
    WeightsTensor(const WeightsTensor&) = default;
    WeightsTensor& operator=(const WeightsTensor&) = default;

    WeightsTensor(const NDims& nd, WeightsType dt, WeightsLayout l, size_t of = 0, size_t sz = 0, float pv = 0.f)
        : TensorBaseT(nd, GetBlocked(l), dt, l, of, sz, pv) {}

    WeightsTensor(const std::vector<size_t>& d, WeightsType dt, WeightsLayout l)
        : TensorBaseT<WeightsType, WeightsLayout>(GetSimpleDims(d, l), GetBlocked(l), dt, l) {}

    WeightsTensor TransformIgnorePadding(WeightsLayout l) const { return TransformIgnorePadding(l, dtype); }
    WeightsTensor TransformIgnorePadding(WeightsLayout l, WeightsType t, size_t g = 1, bool should_split = true) const;

    Dim X() const { return Extract(layout, WeightsChannelName::X, dims); }
    Dim Y() const { return Extract(layout, WeightsChannelName::Y, dims); }
    Dim Z() const { return Extract(layout, WeightsChannelName::Z, dims); }
    Dim IFM() const { return Extract(layout, WeightsChannelName::IFM, dims); }
    Dim OFM() const { return Extract(layout, WeightsChannelName::OFM, dims); }
    Dim LX() const { return Extract(layout, WeightsChannelName::LX, dims); }
    Dim LY() const { return Extract(layout, WeightsChannelName::LY, dims); }
    Dim G() const { return Extract(layout, WeightsChannelName::G, dims); }

    static inline Dim Extract(WeightsLayout l, WeightsChannelName channel, const NDims& d) {
        return TensorBaseT::Extract(weightsChannelArray, l, channel, d);
    }

    static inline int Channelndex(WeightsLayout l, WeightsChannelName channel) {
        return TensorBaseT::ChannelIndex(weightsChannelArray, l, channel);
    }

    static inline bool DoesGroupDimExist(WeightsLayout l) {
        return TensorBaseT::ChannelIndex(weightsChannelArray, l, WeightsChannelName::G) != -1;
    }

    static inline uint32_t ChannelsCount(WeightsLayout l) { return TensorBaseT::ChannelsCount(weightsChannelArray, l); }

    static inline uint32_t BlockedChannelsCount(WeightsLayout l) { return TensorBaseT::BlockedChannelsCount(weightsChannelArray, l); }

    static inline std::vector<int> GetSortedChannelsIndices(WeightsLayout l) {
        return TensorBaseT::GetSortedChannelsIndicesT<WeightsChannelArray, WeightsChannelName>(weightsChannelArray, l);
    }
private:
    struct WeightsChannelDesc {
        std::array<int, static_cast<size_t>(WeightsChannelName::COUNT)> channelsOrder;
        std::vector<std::pair<WeightsChannelName, size_t>> blockedChannels;
    };

    using WeightsChannelArray = std::map<WeightsLayout, WeightsChannelDesc>;
    static WeightsChannelArray weightsChannelArray;
    static NDims GetSimpleDims(const std::vector<size_t>& d, WeightsLayout l);
    static inline BlockSizesDesc GetBlocked(WeightsLayout l) {
        return TensorBaseT::GetBlockedChannels<WeightsChannelArray, WeightsChannelName>(weightsChannelArray, l);
    }
};

inline bool GroupedLayout(WeightsLayout l) {
    return WeightsTensor::DoesGroupDimExist(l);
}

}  // namespace Tensor
}  // namespace kernel_selector
