// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
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
    fbyx,                   // 3D+batch
    bfxy,                   // 3D+batch
    byfx,
    bxfy,
    b_fs_yx_fsv2,
    b_fs_zyx_fsv2,
    b_fs_yx_fsv4,           // reordering format for swizzled input for convolution using IMAD
    b_fs_zyx_fsv4,
    b_fs_yx_fsv8,
    b_fs_zyx_fsv8,
    b_fs_yx_fsv16,          // 3D+batch
    b_fs_zyx_fsv16,         // batch, feature, 3D spatial. Blocks of 16 input channels
    b_fs_yx_fsv32,          // 3D+batch
    b_fs_zyx_fsv32,         // 4D+batch
    bs_fs_yx_bsv16_fsv16,   // batch, feature, 2D spatial. Blocks of 16 batch and channels
    bs_fs_yx_bsv16_fsv32,   // batch, feature, 2D spatial. Blocks of 16 batch and 32 channels
    bs_fs_zyx_bsv16_fsv32,  // batch, feature, 3D spatial. Blocks of 16 batch and 32 channels
    bs_fs_zyx_bsv16_fsv16,  // batch, feature, 3D spatial. Blocks of 16 batch and channels
    bs_fs_yx_bsv4_fsv4,     // batch, feature, 2D spatial. Blocks of 4 batch and 4 channels
    bs_fs_yx_bsv8_fsv4,     // batch, feature, 2D spatial. Blocks of 8 batch and 4 channels
    bs_fs_yx_bsv8_fsv2,     // batch, feature, 2D spatial. Blocks of 8 batch and 2 channels
    bs_fs_zyx_bsv8_fsv4,    // batch, feature, 3D spatial. Blocks of 8 batch and 4 channels
    bs_fs_zyx_bsv8_fsv2,    // batch, feature, 3D spatial. Blocks of 8 batch and 2 channels
    bs_fs_yx_bsv16_fsv8,    // batch, feature, 2D spatial. Blocks of 16 batch and 8 channels
    bs_fs_zyx_bsv16_fsv8,   // batch, feature, 3D spatial. Blocks of 16 batch and 8 channels
    bs_fs_yx_bsv16_fsv4,    // batch, feature, 2D spatial. Blocks of 16 batch and 4 channels
    bs_fs_zyx_bsv16_fsv4,   // batch, feature, 3D spatial. Blocks of 16 batch and 4 channels
    bs_fs_yx_bsv16_fsv2,    // batch, feature, 2D spatial. Blocks of 16 batch and 2 channels
    bs_fs_zyx_bsv16_fsv2,   // batch, feature, 3D spatial. Blocks of 16 batch and 2 channels
    bs_fs_yx_bsv4_fsv2,     // batch, feature, 2D spatial. Blocks of 4 batch and 2 channels
    bs_fs_yx_bsv32_fsv32,   // batch, feature, 2D spatial. Blocks of 32 batch and 32 channels
    bs_fs_yx_bsv32_fsv16,   // batch, feature, 2D spatial. Blocks of 32 batch and 16 channels
    bs_fs_zyx_bsv32_fsv32,  // batch, feature, 3D spatial. Blocks of 32 batch and 32 channels
    bs_fs_zyx_bsv32_fsv16,  // batch, feature, 3D spatial. Blocks of 32 batch and 16 channels
    bs_f_bsv8__af8,         // for optimized FC
    bs_f_bsv16__af8,        // for optimized FC
    winograd_2x3_s1_data,   // winograd convolution input, F(2,3) -- filter 3x3 with stride 1
    bfzyx,                  // batch+feature+3D spatial
    bzyxf,
    fs_b_yx_fsv32,          // for FP16 kernels, 32 features to avoid partial writes
    bfwzyx,                 // batch, feature, 4D spatial
    bfuwzyx,                // batch, feature, 5D spatial
    bfvuwzyx,               // batch, feature, 6D spatial
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
    ioyx,
    oyxi,
    oyix,
    oxiy,
    iyxo,
    yxio,
    o_is_yx_isv4,
    o_is_yx_isv16,
    os_iyx_osv16,
    os_iyx_osv32,
    os_iyx_osv8,
    os_iyx_osv32__ai32,
    os_iyx_osv64,
    os_is_zyx_isv16_osv16,
    is_os_zyx_isv16_osv16,
    is_os_yx_isv16_osv16,
    os_is_zyx_isv8_osv16_isv2,
    os_is_yx_isv8_osv16_isv2,
    os_is_yx_isv16_osv16,
    os_zyxi_osv16,
    os_iyx_osv16_rotate_180,
    os_i_osv8__ai8,  // TODO can we drop the alignment form layout name?
    os_i_osv16__ai8,
    os_i_osv16,
    os_is_yx_osv16_isv16,           // weights for int8 blocked conv
    os_is_yx_osv32_isv2,            // weights for fully connected kernels with int4 compressed data type
    os_is_yx_osv64_isv2,            // weights for fully connected kernels with int4 compressed data type
    os_is_zyx_osv16_isv16,
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
    os_is_yx_isa8_osv8_isv4,                 // for MMAD convolution
    os_is_zyx_isa8_osv8_isv4,                // for MMAD convolution
    os_is_yx_isa8_osv16_isv4,                // for fully connected MMAD
    os_is_zyx_isa8_osv16_isv4,               // for fully connected MMAD
    os_is_yx_osa4_isa8_osv8_isv4,            // for MMAD convolution swizzled from ofm 0..7 to 0,4,8,12,16,20,24,28,
    os_is_zyx_osa4_isa8_osv8_isv4,           // for MMAD convolution swizzled from ofm 0..7 to 0,4,8,12,16,20,24,28,
    os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,  // for MMAD convolution swizzled from ofm 0..7 to 0,4,8,12,16,20,24,28,
                                                 // 1,5...
    os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4,  // for MMAD convolution swizzled from ofm 0..7 to 0,4,8,12,16,20,24,28,
                                                  // 1,5...
    os_is_yx_osv16_isv4,                 // swizzled weights for convolution using IMAD
    os_is_yx_osv8_isv4,                      // weights for int8 blocked conv
    os_is_yx_osv32_isv4_swizzled_by_2,   //  weights for bfyx -> b_fs_yx_fsv32 convolution using IMAD with swizzled ofm (0, 2, 4..), (1, 3, 5...)
    os_is_yx_osv32_isv4,                 //  weights for bfyx -> b_fs_yx_fsv{32,16} convolution using IMAD
    os_is_zyx_osv32_isv4,                //  weights for bfzyx -> b_fs_zyx_fsv16 convolution using IMAD
    oizyx,
    iozyx,
    goiyx,
    gioyx,
    goizyx,
    giozyx,
    gyxio,
    g_os_iyx_osv8,
    g_os_iyx_osv16,
    g_os_iyx_osv32,
    gs_oiyx_gsv16,
    gs_oizyx_gsv16,
    gs_oiyx_gsv32,
    g_os_iyx_osv16_rotate_180,
    gi_yxs_os_yxsv2_osv16,
    g_is_os_zyx_isv16_osv16,
    g_is_os_yx_isv16_osv16,
    g_os_is_zyx_isv8_osv16_isv2,
    g_os_is_yx_isv8_osv16_isv2,
    g_os_is_zyx_isv16_osv16,
    g_os_is_zyx_osv16_isv16,
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
    bool is_dynamic = false; // Currently cannot set pad_before and pad_after as dynamic separately

    Pad(size_t before, size_t after, bool is_dynamic = false) : before(before), after(after), is_dynamic(is_dynamic) {}

    static size_t NumPadOffsetsPerDim() { return 2; /*pad_before/pad_after*/}
    size_t Total() const {
        OPENVINO_ASSERT(!is_dynamic, "Total() is called for dynamic pad!");
        return before + after;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dim
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Dim {
    size_t v;
    size_t pitch;
    Pad pad;
    bool is_dynamic;

    Dim(size_t v = 0, size_t pitch = 0, Pad pad = {0, 0, false}, bool is_dynamic = false)
        : v(v),
          pitch(pitch),
          pad(pad),
          is_dynamic(is_dynamic) {}

    size_t LogicalDimPadded() const {
        OPENVINO_ASSERT(!pad.is_dynamic, "LogicalDimPadded() is called for dynamic pad");
        return v + pad.Total();
    }
};

using NDims = std::vector<Dim>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extract code
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class DataChannelName { X = 0, Y = 1, Z = 2, W = 3, U = 4, V = 5, FEATURE = 6, BATCH = 7, COUNT = 8 };

enum class WeightsChannelName { X = 0, Y = 1, Z = 2, IFM = 3, OFM = 4, G = 5, COUNT = 6 };

inline bool SimpleLayout(WeightsLayout l) {
    switch (l) {
        case WeightsLayout::oi:
        case WeightsLayout::io:
        case WeightsLayout::oiyx:
        case WeightsLayout::ioyx:
        case WeightsLayout::oyxi:
        case WeightsLayout::oyix:
        case WeightsLayout::oxiy:
        case WeightsLayout::iyxo:
        case WeightsLayout::yxio:
        case WeightsLayout::oizyx:
        case WeightsLayout::iozyx:
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
        case DataLayout::byfx:
        case DataLayout::bxfy:
        case DataLayout::fbyx:
        case DataLayout::fyxb:
        case DataLayout::bfxy:
        case DataLayout::bfzyx:
        case DataLayout::bzyxf:
        case DataLayout::bfwzyx:
        case DataLayout::bfuwzyx:
        case DataLayout::bfvuwzyx:
            return true;
        default:
            return false;
    }
}

inline bool DoubleBlockedLayout(DataLayout l) {
    switch (l) {
        case DataLayout::bs_fs_yx_bsv16_fsv32:
        case DataLayout::bs_fs_yx_bsv16_fsv16:
        case DataLayout::bs_fs_zyx_bsv16_fsv32:
        case DataLayout::bs_fs_zyx_bsv16_fsv16:
        case DataLayout::bs_fs_yx_bsv4_fsv4:
        case DataLayout::bs_fs_yx_bsv8_fsv4:
        case DataLayout::bs_fs_yx_bsv8_fsv2:
        case DataLayout::bs_fs_zyx_bsv8_fsv4:
        case DataLayout::bs_fs_zyx_bsv8_fsv2:
        case DataLayout::bs_fs_yx_bsv16_fsv4:
        case DataLayout::bs_fs_zyx_bsv16_fsv4:
        case DataLayout::bs_fs_yx_bsv16_fsv2:
        case DataLayout::bs_fs_zyx_bsv16_fsv2:
        case DataLayout::bs_fs_yx_bsv4_fsv2:
        case DataLayout::bs_fs_yx_bsv32_fsv32:
        case DataLayout::bs_fs_yx_bsv32_fsv16:
        case DataLayout::bs_fs_zyx_bsv32_fsv32:
        case DataLayout::bs_fs_zyx_bsv32_fsv16:
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tensor Explanation
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
    size_t viewOffset = 0;  // in elements
    size_t firstElementOffset = 0;
    size_t totalSize = 0;  // in elements
    float paddedVal = 0.f;

public:
    TensorBase() = default;
    TensorBase(const TensorBase&) = default;
    TensorBase& operator=(const TensorBase&) = default;

    TensorBase(const NDims& nd, size_t viewOf, size_t sz, float pv)
        : dims(nd),
          viewOffset(viewOf),
          firstElementOffset(std::accumulate(nd.cbegin(),
                                             nd.cend(),
                                             viewOf,
                                             [](size_t val, const Dim& d) { return val + d.pitch * d.pad.before; })),
          totalSize(sz),
          paddedVal(pv) {
        if (!std::any_of(dims.begin(), dims.end(), [](const Dim& d) {
                return d.pad.is_dynamic;
            })) {
            if (totalSize == 0) {
                for (const auto& d : dims) {
                    totalSize = std::max(totalSize, d.pitch * (d.LogicalDimPadded()));
                }

                totalSize += viewOffset;
            }

            size_t minimalPitch = 1;

            for (const auto& d : dims) {
                if (d.pitch < minimalPitch) {
                    throw std::runtime_error("Tensor pitches didn't set correctly");
                }

                minimalPitch *= d.LogicalDimPadded();
            }

            if (totalSize < (minimalPitch + viewOffset)) {
                throw std::runtime_error("Tensor total Size didn't set correctly");
            }
        }
    }

    float GetPaddedVal() const { return paddedVal; }
    size_t GetFirstElementOffset() const { return firstElementOffset; }
    size_t GetViewOffset() const { return viewOffset; }
    const NDims& GetDims() const { return dims; }

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

        size_t calc_pitch = 1;
        for (const auto& d : dims) {
            differ |= (d.pitch != calc_pitch);
            calc_pitch *= d.v;
        }

        return differ;
    }

    bool is_dynamic() const {
        return std::any_of(dims.begin(), dims.end(), [](const Dim& d) { return d.is_dynamic; });
    }

    bool has_dynamic_pad() const {
        return std::any_of(dims.begin(), dims.end(), [](const Dim& d) { return d.pad.is_dynamic; });
    }

    virtual ~TensorBase() = default;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TensorBaseT
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DType, typename Layout>
struct TensorBaseT : public TensorBase {
protected:
    DType dtype = DType();
    Layout layout = Layout();

    template <typename ArrayT, typename ChannelName>
    static inline int ChannelIndex(const ArrayT& channelArr, Layout l, ChannelName channelName) {
        size_t channel = static_cast<size_t>(channelName);

        for (auto& entry : channelArr) {
            if (entry.first == l)
                return entry.second[channel];
        }

        return -1;
    }

    template <typename ArrayT, typename ChannelName>
    static inline Dim Extract(const ArrayT& channelArr, Layout l, ChannelName channelName, const NDims& dims) {
        const int i = ChannelIndex(channelArr, l, channelName);
        return ((i < 0) || (i >= static_cast<int>(dims.size()))) ? Dim{1, 1, Pad{0, 0, false}} : dims[i];
    }

    template <typename ArrayT>
    static inline uint32_t ChannelsCount(const ArrayT& channelArr, Layout l) {
        const auto& entry =
            std::find_if(std::begin(channelArr),
                         std::end(channelArr),
                         [&](typename std::tuple_element<0, ArrayT>::type entry) { return entry.first == l; });

        if (entry == channelArr.end())
            throw std::invalid_argument("Failed to get channels count for layout " +
                                        std::to_string(static_cast<uint32_t>(l)));

        return std::accumulate(entry->second.begin(), entry->second.end(), 0U, [](uint32_t count, int v) {
            return count + ((v != -1) ? 1 : 0);
        });
    }

public:
    TensorBaseT() = default;
    TensorBaseT(const TensorBaseT&) = default;
    TensorBaseT& operator=(const TensorBaseT&) = default;

    TensorBaseT(const NDims& nd, DType dt, Layout l, size_t of = 0, size_t sz = 0, float pv = 0.f)
        : TensorBase(nd, of, sz, pv), dtype(dt), layout(l) {}

    DType GetDType() const { return dtype; }
    Layout GetLayout() const { return layout; }
    uint32_t ElementSize() const override { return BytesPerElement(dtype); }
    size_t Dimentions() const { return dims.size(); }
    bool SimpleLayout() const { return Tensor::SimpleLayout(layout); }
    bool DoubleBlockedLayout() const { return Tensor::DoubleBlockedLayout(layout); }
    bool GroupedLayout() const { return Tensor::GroupedLayout(layout); }

    bool operator==(const TensorBaseT& t) const {
        bool same = dtype == t.dtype && layout == t.layout && paddedVal == t.paddedVal && viewOffset == t.viewOffset &&
                    dims.size() == t.dims.size();
        if (same) {
            for (size_t i = 0; i < dims.size(); i++) {
                same &= dims[i].v == t.dims[i].v && dims[i].pad.before == t.dims[i].pad.before &&
                        dims[i].pad.after == t.dims[i].pad.after && dims[i].pitch == t.dims[i].pitch &&
                        dims[i].pad.is_dynamic == t.dims[i].pad.is_dynamic;
            }
        }

        return same;
    }

    bool operator!=(const TensorBaseT& t) const {
        return !(*this == t);
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
        : TensorBaseT(nd, dt, l, of, sz, pv) {}

    DataTensor(const std::vector<size_t>& d, Datatype dt, DataLayout l)
        : TensorBaseT<Datatype, DataLayout>(GetSimpleDims(d, l), dt, l) {}

    Dim X() const { return Extract(layout, DataChannelName::X, dims); }
    Dim Y() const { return Extract(layout, DataChannelName::Y, dims); }
    Dim Z() const { return Extract(layout, DataChannelName::Z, dims); }
    Dim W() const { return Extract(layout, DataChannelName::W, dims); }
    Dim U() const { return Extract(layout, DataChannelName::U, dims); }
    Dim V() const { return Extract(layout, DataChannelName::V, dims); }
    Dim Feature() const { return Extract(layout, DataChannelName::FEATURE, dims); }
    Dim Batch() const { return Extract(layout, DataChannelName::BATCH, dims); }

    DataTensor TransformIgnorePadding(DataLayout l) const;
    DataTensor FlattenFeatureAndSpatials() const;
    DataTensor FlattenEverything() const;
    void SwapXY();
    void SetDynamicShapeOffset(size_t offset) {
        dynamic_shape_offset = offset;
    }

    size_t get_dynamic_shape_offset() const {
        return dynamic_shape_offset;
    }

    static inline Dim Extract(DataLayout l, DataChannelName channel, const NDims& d) {
        return TensorBaseT::Extract(dataChannelArray, l, channel, d);
    }

    static inline int Channelndex(DataLayout l, DataChannelName channel) {
        return TensorBaseT::ChannelIndex(dataChannelArray, l, channel);
    }

    static inline uint32_t ChannelsCount(DataLayout l) { return TensorBaseT::ChannelsCount(dataChannelArray, l); }

    static size_t max_rank() { return static_cast<size_t>(DataChannelName::COUNT); }

private:
    using DataChannelDesc = std::pair<DataLayout, std::array<int, static_cast<size_t>(DataChannelName::COUNT)>>;
    using DataChannelArray = std::array<DataChannelDesc, DataLayout::DataLayoutCount>;
    static DataChannelArray dataChannelArray;
    static NDims GetSimpleDims(const std::vector<size_t>& d, DataLayout l);
    size_t dynamic_shape_offset = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WeightsTensor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct WeightsTensor : TensorBaseT<WeightsType, WeightsLayout> {
    WeightsTensor() = default;
    WeightsTensor(const WeightsTensor&) = default;
    WeightsTensor& operator=(const WeightsTensor&) = default;

    WeightsTensor(const NDims& nd, WeightsType dt, WeightsLayout l, size_t of = 0, size_t sz = 0, float pv = 0.f)
        : TensorBaseT(nd, dt, l, of, sz, pv) {}

    WeightsTensor(const std::vector<size_t>& d, WeightsType dt, WeightsLayout l)
        : TensorBaseT<WeightsType, WeightsLayout>(GetSimpleDims(d, l), dt, l) {}

    WeightsTensor TransformIgnorePadding(WeightsLayout l) const { return TransformIgnorePadding(l, dtype); }
    WeightsTensor TransformIgnorePadding(WeightsLayout l, WeightsType t, size_t g = 1, bool should_split = true) const;

    Dim X() const { return Extract(layout, WeightsChannelName::X, dims); }
    Dim Y() const { return Extract(layout, WeightsChannelName::Y, dims); }
    Dim Z() const { return Extract(layout, WeightsChannelName::Z, dims); }
    Dim IFM() const { return Extract(layout, WeightsChannelName::IFM, dims); }
    Dim OFM() const { return Extract(layout, WeightsChannelName::OFM, dims); }
    Dim G() const { return Extract(layout, WeightsChannelName::G, dims); }

    void SwapXY();

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

private:
    using WeightsChannelDesc =
        std::pair<WeightsLayout, std::array<int, static_cast<size_t>(WeightsChannelName::COUNT)>>;
    using WeightsChannelArray = std::array<WeightsChannelDesc, WeightsLayout::WeightsLayoutCount>;
    static WeightsChannelArray weightsChannelArray;
    static NDims GetSimpleDims(const std::vector<size_t>& d, WeightsLayout l);
};

inline bool GroupedLayout(WeightsLayout l) {
    return WeightsTensor::DoesGroupDimExist(l);
}

}  // namespace Tensor
}  // namespace kernel_selector
