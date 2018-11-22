/*
// Copyright (c) 2016 Intel Corporation
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
*/

#pragma once

#include "common_types.h"
#include <vector>
#include <assert.h>
#include <numeric>
#include <cstddef>
#include <algorithm>
#include <array>

namespace kernel_selector
{
#define KERNEL_SELECTOR_TENSOR_DIM_MAX 8

    namespace Tensor
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // DataLayout
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        enum DataLayout
        {
            bf = 0,             // 1D+batch
            fb,                 // 1D+batch
            bfyx,               // 3D+batch
            yxfb,               // 3D+batch
            byxf,               // 3D+batch
            fyxb,               // 3D+batch
            bs_f_bsv8__af8,     // for optimized FC
            bs_f_bsv16__af8,    // for optimized FC
            bf8_xy16,           // for optimized conv1x1
            // TODO: most of the kernel doesn't support ROI. we need to handle it correctly.
            brfyx,              // 4D+batch
            winograd_2x3_s1_data, //winograd convolution input, F(2,3) -- filter 3x3 with stride 1
            byxf_af32, // for MMAD convolution
            fs_bs_yx_bsv4_fsv32, // for batched MMAD
            DataLayoutCount // NMBER OF ELEMENTS IN ENUM
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // WeightsLayout
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        enum WeightsLayout
        {
            oi = 0,
            io,
            oiyx,
            oyxi,
            iyxo,
            yxio,
            os_iyx_osv16,
            os_iyx_osv16_rotate_180,
            os_i_osv16,
            os_i_osv8__ai8,         // TODO can we drop the alignment form layout name?
            os_i_osv16__ai8,
            i_yxs_os_yxsv2_osv16,
            iy_xs_os_xsv2_osv16__ao32,
            iy_xs_os_xsv2_osv8__ao32,
            image_2d_weights_c4_fyx_b,           // image type fyx_b
            image_2d_weights_c1_b_fyx,     // image type 2d b_fyx single channel
            winograd_2x3_s1_weights, //winograd convolution weights, F(2, 3) --filter 3x3 with stride 1
            winograd_2x3_s1_fused_weights, //winograd convolution weights for fused kernel, F(2, 3) --filter 3x3 with stride 1
            winograd_6x3_s1_fused_weights, //winograd convolution weights for fused kernel, F(6, 3) --filter 3x3 with stride 1
            image_2d_weights_winograd_6x3_s1_fbxyb, // image 2d winograd convolution weights for fused kernel, F(2, 3) --filter 3x3 with stride 1
            image_2d_weights_winograd_6x3_s1_xfbyb, // image 2d winograd convolution weights for fused kernel, F(2, 3) --filter 3x3 with stride 1
            os_is_yx_isa8_osv8_isv4, // for MMAD convolution
            is_o_yx_isv32, // for MMAD 1x1 convolutions
            WeightsLayoutCount // NMBER OF ELEMENTS IN ENUM
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pad
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct Pad
        {
            size_t before;
            size_t after;

            size_t Total() const { return before + after; }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Dim
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct Dim
        {
            size_t v;
            size_t pitch;
            Pad    pad;

            size_t LogicalDimPadded() const { return v + pad.Total(); }
        };

        using NDims = std::vector<Dim>;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // extract code
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        enum class DataChannelName
        {
            X       = 0,
            Y       = 1,
            FEATURE = 2,
            ROI     = 3,
            BATCH   = 4,
        };

        enum class WeightsChannelName
        {
            X   = 0,
            Y   = 1,
            IFM = 2,
            OFM = 3,
        };

        inline bool SimpleLayout(WeightsLayout l)
        {
            switch (l)
            {
            case WeightsLayout::oi:
            case WeightsLayout::io:
            case WeightsLayout::oiyx:
            case WeightsLayout::oyxi:
            case WeightsLayout::iyxo:
            case WeightsLayout::yxio:
                return true;
            default:
                return false;
            }
        }

        inline bool SimpleLayout(DataLayout l)
        {
            switch (l)
            {
            case DataLayout::bf:
            case DataLayout::fb:
            case DataLayout::bfyx:
            case DataLayout::yxfb:
            case DataLayout::byxf:
            case DataLayout::fyxb:
                return true;
            default:
                return false;
            }
        }

        inline bool IsImageType(WeightsLayout l)
        {
            switch (l)
            {
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
        struct TensorBase
        {
        protected:
            NDims   dims;
            size_t  viewOffset          = 0;    // in elements
            size_t  firstElementOffset  = 0;
            size_t  totalSize           = 0;    // in elements
            float   paddedVal           = 0.f;

        public:
            TensorBase() = default;
            TensorBase(const TensorBase&) = default;
            TensorBase& operator=(const TensorBase&) = default;

            TensorBase(const NDims& nd, size_t viewOf, size_t sz, float pv)
                : dims(nd)
                , viewOffset(viewOf)
                , firstElementOffset(std::accumulate(nd.cbegin(), nd.cend(), viewOf, [](size_t val, const Dim& d) { return val + d.pitch*d.pad.before; }))
                , totalSize(sz)
                , paddedVal(pv)
            {
                if (totalSize == 0)
                {
                    for (const auto& d : dims)
                    {
                        totalSize = std::max(totalSize, d.pitch*(d.LogicalDimPadded()));
                    }

                    totalSize += viewOffset;
                }

                size_t minimalPitch = 1;

                for (const auto& d : dims)
                {
                    if (d.pitch < minimalPitch)
                    {
                        throw std::runtime_error("Tensor pitches didn't set correctly");
                    }

                    minimalPitch *= d.LogicalDimPadded();
                }

                if (totalSize < (minimalPitch + viewOffset))
                {
                    throw std::runtime_error("Tensor total Size didn't set correctly");
                }
            }

            float           GetPaddedVal()          const { return paddedVal; }
            size_t          GetFirstElementOffset() const { return firstElementOffset; }
            size_t          GetViewOffset()         const { return viewOffset; }
            const NDims&    GetDims()               const { return dims; }

            virtual uint32_t    ElementSize() const = 0;

            // Size of the actual data (without padded part)
            size_t LogicalSize() const
            {
                return std::accumulate(dims.cbegin(), dims.cend(), (size_t)1, [](size_t val, const Dim& d) {return val*d.v; });
            }

            // Dimensions of the actual data (without padded part)
            std::vector<size_t> LogicalDims() const
            {
                std::vector<size_t> res(dims.size());
                std::transform(dims.begin(), dims.end(), res.begin(), [](const Dim& d) { return d.v; });
                return res;
            }

            // Whole buffer size (in elements)
            size_t PhysicalSize() const
            {
                return totalSize;
            }

            // Whole buffer size (in bytes)
            size_t PhysicalSizeInBytes() const
            {
                return totalSize * ElementSize();
            }

            // if padded/view exists between logical dimensions.
            // in other words, if we can consider the data as a 1Dim resource.
            bool PitchesDifferFromLogicalDims() const
            {
                bool differ = false;

                size_t calc_pitch = 1;
                for (const auto& d : dims)
                {
                    differ |= (d.pitch != calc_pitch);
                    calc_pitch *= d.v;
                }
                
                return differ;
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // TensorBaseT
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template<typename DType, typename Layout>
        struct TensorBaseT : public TensorBase
        {
        protected:
            DType     dtype;
            Layout    layout;

            template <typename ArrayT, typename ChannelName>
            static inline int Channelndex(const ArrayT& channelArr, Layout l, ChannelName channelName)
            {
                size_t channel = static_cast<size_t>(channelName);
                assert(channel < channelArr[l].size());

                return channelArr[l][channel];
            }

            template <typename ArrayT, typename ChannelName>
            static inline Dim Extract(const ArrayT& channelArr, Layout l, ChannelName channelName, const NDims& dims)
            {
                const int i = Channelndex(channelArr, l, channelName);
                return ((i < 0) || (i >= (int)dims.size())) ? Dim{ 1, 1,{ 0,0 } } : dims[i];
            }

            template <typename ArrayT>
            static inline uint32_t ChannelsCount(const ArrayT& channelArr, Layout l)
            {
                const auto& entry = channelArr[l];
                return std::accumulate(entry.begin(), entry.end(), 0U, [](uint32_t count, int v) {return count + ((v != -1) ? 1 : 0); });
            }

        public:
            TensorBaseT() = default;
            TensorBaseT(const TensorBaseT&) = default;
            TensorBaseT& operator=(const TensorBaseT&) = default;

            TensorBaseT(const NDims& nd, DType dt, Layout l, size_t of = 0, size_t sz = 0, float pv = 0.f) :
                TensorBase(nd, of, sz, pv), dtype(dt), layout(l) {}

            DType       GetDType()      const           { return dtype; }
            Layout      GetLayout()     const           { return layout; }
            uint32_t    ElementSize()   const override  { return BytesPerElement(dtype); }
            size_t      Dimentions()    const           { return dims.size(); }
            bool        SimpleLayout()  const           { return Tensor::SimpleLayout(layout); }

            bool operator==(const TensorBaseT& t) const
            {
                bool same =
                    dtype == t.dtype      &&
                    layout == t.layout     &&
                    paddedVal == t.paddedVal  &&
                    viewOffset == t.viewOffset     &&
                    dims.size() == t.dims.size();
                if (same)
                {
                    for (size_t i = 0; i < dims.size(); i++)
                    {
                        same &=
                            dims[i].v == t.dims[i].v &&
                            dims[i].pad.before == t.dims[i].pad.before &&
                            dims[i].pad.after == t.dims[i].pad.after &&
                            dims[i].pitch == t.dims[i].pitch;
                    }
                }

                return same;
            }

            bool SameDims(const TensorBaseT& t) const
            {
                bool same =
                    dtype == t.dtype &&
                    layout == t.layout &&
                    dims.size() == t.dims.size();
                if (same)
                {
                    for (size_t i = 0; i < dims.size(); i++)
                    {
                        same &= dims[i].v == t.dims[i].v;
                    }
                }

                return same;
            }

            bool SameDimsSizes(const TensorBaseT& t) const
            {
                bool same =
                    dims.size() == t.dims.size();
                if (same)
                {
                    for (size_t i = 0; i < dims.size(); i++)
                    {
                        same &= dims[i].v == t.dims[i].v;
                    }
                }

                return same;
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // DataTensor
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct DataTensor : public TensorBaseT<Datatype, DataLayout>
        {
            DataTensor() = default;
            DataTensor(const DataTensor&) = default;
            DataTensor& operator=(const DataTensor&) = default;

            DataTensor(const NDims& nd, Datatype dt, DataLayout l, size_t of = 0, size_t sz = 0, float pv = 0.f) :
                TensorBaseT(nd, dt, l, of, sz, pv) {}

            DataTensor(const std::vector<size_t>& d, Datatype dt, DataLayout l) :
                TensorBaseT<Datatype, DataLayout>(GetSimpleDims(d, l), dt, l) {}

            Dim X()         const { return Extract(layout, DataChannelName::X, dims); }
            Dim Y()         const { return Extract(layout, DataChannelName::Y, dims); }
            Dim Feature()   const { return Extract(layout, DataChannelName::FEATURE, dims); }
            Dim ROI()       const { return Extract(layout, DataChannelName::ROI, dims); }
            Dim Batch()     const { return Extract(layout, DataChannelName::BATCH, dims); }

            DataTensor  TransformIgnorePadding(DataLayout l) const;
            DataTensor  FlattenFeatureAndSpatials() const;

            static inline Dim Extract(DataLayout l, DataChannelName channel, const NDims& d)
            {
                return TensorBaseT::Extract(dataChannelArray, l, channel, d);
            }

            static inline int Channelndex(DataLayout l, DataChannelName channel)
            {
                return TensorBaseT::Channelndex(dataChannelArray, l, channel);
            }

            static inline uint32_t ChannelsCount(DataLayout l)
            {
                return TensorBaseT::ChannelsCount(dataChannelArray, l);
            }
        private:
            static std::array<std::array<int, 5>, DataLayout::DataLayoutCount> dataChannelArray;
            static NDims GetSimpleDims(const std::vector<size_t>& d, DataLayout l);
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // WeightsTensor
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        struct WeightsTensor : TensorBaseT<WeightsType, WeightsLayout>
        {
            WeightsTensor() = default;
            WeightsTensor(const WeightsTensor&) = default;
            WeightsTensor& operator=(const WeightsTensor&) = default;

            WeightsTensor(const NDims& nd, WeightsType dt, WeightsLayout l, size_t of = 0, size_t sz = 0, float pv = 0.f) :
                TensorBaseT(nd, dt, l, of, sz, pv) {}

            WeightsTensor(const std::vector<size_t>& d, WeightsType dt, WeightsLayout l) :
                TensorBaseT<WeightsType, WeightsLayout>(GetSimpleDims(d, l), dt, l) {}

            WeightsTensor TransformIgnorePadding(WeightsLayout l) const { return TransformIgnorePadding(l, dtype); }
            WeightsTensor TransformIgnorePadding(WeightsLayout l, WeightsType t) const;

            Dim X()   const { return Extract(layout, WeightsChannelName::X, dims); }
            Dim Y()   const { return Extract(layout, WeightsChannelName::Y, dims); }
            Dim IFM() const { return Extract(layout, WeightsChannelName::IFM, dims); }
            Dim OFM() const { return Extract(layout, WeightsChannelName::OFM, dims); }

            static inline Dim Extract(WeightsLayout l, WeightsChannelName channel, const NDims& d)
            {
                return TensorBaseT::Extract(weightsChannelArray, l, channel, d);
            }

            static inline int Channelndex(WeightsLayout l, WeightsChannelName channel)
            {
                return TensorBaseT::Channelndex(weightsChannelArray, l, channel);
            }

            static inline uint32_t ChannelsCount(WeightsLayout l)
            {
                return TensorBaseT::ChannelsCount(weightsChannelArray, l);
            }
        private:
            static NDims GetSimpleDims(const std::vector<size_t>& d, WeightsLayout l);
            static std::array<std::array<int, 4>, WeightsLayout::WeightsLayoutCount> weightsChannelArray;
        };
    }
}