/*
// Copyright (c) 2016-2019 Intel Corporation
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

#include <cstddef>
#include "tensor_type.h"
#include "common_tools.h"

namespace kernel_selector
{
    namespace Tensor
    {
        std::array<std::array<int, 5>, DataLayout::DataLayoutCount> DataTensor::dataChannelArray
        { {
            // explaination:
            // 0, 1, 2, 3, 4 means the ordering starts from X, then Y, then F, thenR, then B
            // -1 means it's not used
            //X, Y, F, R, B
            {-1,-1, 0,-1, 1 }, // DataLayout::bf
            {-1,-1, 1,-1, 0 }, // DataLayout::fb
            { 0, 1, 2,-1, 3 }, // DataLayout::bfyx
            { 2, 3, 1,-1, 0 }, // DataLayout::yxfb
            { 1, 2, 0,-1, 3 }, // DataLayout::byxf
            { 1, 2, 3,-1, 0 }, // DataLayout::fyxb
            {-1,-1, 0,-1, 1 }, // DataLayout::bs_f_bsv8__af8
            {-1,-1, 0,-1, 1 }, // DataLayout::bs_f_bsv16__af8
            { 0, 1, 2,-1, 3 }, // DataLayout::bf8_xy16
            { 0, 1, 2, 3, 4 }, // DataLayout::brfyx
            { 2, 1, 0,-1, 3 }, // DataLayout::winograd_2x3_s1_data
            { 1, 2, 0,-1, 3 }, // DataLayout::byxf_af32
            { 1, 2, 0,-1, 3 }, // DataLayout::byx8_f8
            { 0, 1, 3,-1, 2 }, // DataLayout::fs_bs_yx_bsv4_fsv32
            { 0, 1, 2, -1, 3 },// DataLayout::b_fs_yx_fsv4
        } };

        std::array<std::array<int, 6>, WeightsLayout::WeightsLayoutCount> WeightsTensor::weightsChannelArray
        { {
            // X, Y,   I,  O, LX, LY,
            { -1, -1,  0,  1, -1, -1 }, // WeightsLayout::oi
            { -1, -1,  1,  0, -1, -1 }, // WeightsLayout::io
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::oiyx
            {  1,  2,  0,  3, -1, -1 }, // WeightsLayout::oyxi
            {  1,  2,  3,  0, -1, -1 }, // WeightsLayout::iyxo
            {  2,  3,  1,  0, -1, -1 }, // WeightsLayout::yxio
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::os_iyx_osv16
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::os_iyx_osv32
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::os_iyx_osv64
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::os_iyx_osv16_rotate_180
            { -1, -1,  0,  1, -1, -1 }, // WeightsLayout::os_i_osv8__ai8
            { -1, -1,  0,  1, -1, -1 }, // WeightsLayout::os_i_osv16__ai8
            { -1, -1,  0,  1, -1, -1 }, // WeightsLayout::os_i_osv16
            {  1,  2,  3,  0, -1, -1 }, // WeightsLayout::i_yxs_os_yxsv2_osv16
            {  1,  2,  3,  0, -1, -1 }, // WeightsLayout::iy_xs_os_xsv2_osv16__ao32
            {  1,  2,  3,  0, -1, -1 }, // WeightsLayout::iy_xs_os_xsv2_osv8__ao32
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::image_2d_weights_c4_fyx_b
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::image_2d_weights_c1_b_fyx
            {  3,  2,  1,  0, -1, -1 }, // WeightsLayout::winograd_2x3_s1_weights
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::winograd_2x3_s1_fused_weights
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::winograd_6x3_s1_fused_weights
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::os_is_yx_isa8_osv8_isv4
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::os_is_yx_isa8_osv8_isv4_swizzled_by_4
            {  1,  2,  0,  3, -1, -1 }, // WeightsLayout::is_o_yx_isv32
            {  1,  2,  0,  3, -1, -1 }, // WeightsLayout::is_o32_yx_isv32_swizzled_by_4
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::os_is_y_x8_osv8_isv4
            {  0,  1,  2,  3,  4,  5 }, // WeightsLayout::bf_lyx_yx
            {  0,  1,  2,  3, -1, -1 }, // WeightsLayout::os_is_yx_osv16_isv4
        } };

        NDims DataTensor::GetSimpleDims(const std::vector<size_t>& d, DataLayout l)
        {
            std::vector<size_t> newDims = d;

            // TOOD: it's not the right pitches. it's here in order to calculate physical size
            switch (l)
            {
            case bs_f_bsv8__af8:
                assert(newDims.size() == 2);
                newDims[0] = RoundUp(newDims[0], 8);
                newDims[1] = RoundUp(newDims[1], 8);
                break;
            case bs_f_bsv16__af8:
                assert(newDims.size() == 2);
                newDims[0] = RoundUp(newDims[0], 8);
                newDims[1] = RoundUp(newDims[1], 16);
                break;
            case bf8_xy16:
                assert(newDims.size() == 4);
                newDims[1] = RoundUp(newDims[1], 8);
                newDims[3] = RoundUp(newDims[2] * newDims[3], 16);
                newDims[2] = 1;
                break;
            case byxf_af32:
                assert(newDims.size() == 4);
                newDims[0] = RoundUp(newDims[0], 32);
                break;
            case byx8_f4:
                assert(newDims.size() == 4);
                newDims[0] = RoundUp(newDims[0], 4);
                newDims[1] = RoundUp(newDims[1], 8);
                break;
            case fs_bs_yx_bsv4_fsv32:
                assert(newDims.size() == 4);
                newDims[3] = RoundUp(newDims[3], 32);
                newDims[2] = RoundUp(newDims[2], 4);
                break;
            default:
                break;
            }

            NDims ret(newDims.size());
            size_t pitch = 1;

            for (size_t i = 0; i < newDims.size(); i++)
            {
                Pad p = { 0, newDims[i] - d[i] };
                ret[i] = { d[i], pitch, p };
                pitch *= newDims[i];
            }

            if (l == byxf_af32 || l == fs_bs_yx_bsv4_fsv32 || l == byx8_f4)
            {
                ret[0].pitch = 1;
                ret[1].pitch = ret[0].pitch * newDims[0];
                ret[2].pitch = ret[1].pitch * newDims[1];
                ret[3].pitch = ret[2].pitch * newDims[2];
                ret[4].pitch = ret[3].pitch * newDims[3];
            }

            return ret;
        }

        DataTensor DataTensor::TransformIgnorePadding(DataLayout l) const
        {
            const uint32_t src_channels = ChannelsCount(layout);
            const uint32_t dst_channels = ChannelsCount(l);

            const size_t src_x = X().v;
            const size_t src_y = Y().v;

            std::vector<size_t> vec(dst_channels);
            if (src_channels == 2 && dst_channels == 2)
            {
                vec[Channelndex(l, DataChannelName::FEATURE)] = Feature().v;
                vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
            }
            else if (src_channels == 4 && dst_channels == 4)
            {
                vec[Channelndex(l, DataChannelName::X)] = X().v;
                vec[Channelndex(l, DataChannelName::Y)] = Y().v;
                vec[Channelndex(l, DataChannelName::FEATURE)] = Feature().v;
                vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
            }
            else if (src_channels == 2 && dst_channels == 4)
            {
                const size_t dst_ifm = Feature().v / (src_x*src_y);
                const size_t dst_xy = Feature().v % (src_x*src_y);
                const size_t dst_y = dst_xy / src_x;
                const size_t dst_x = dst_xy % src_x;
                vec[Channelndex(l, DataChannelName::X)] = dst_x;
                vec[Channelndex(l, DataChannelName::Y)] = dst_y;
                vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
                vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
            }
            else if (src_channels == 4 && dst_channels == 2)
            {
                const size_t dst_ifm = Feature().v * src_x * src_y;
                vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
                vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
            }
            else
            {
                // TODO: implement ROI
                assert(0);
            }

            return{ vec, dtype, l };
        }

        DataTensor DataTensor::FlattenFeatureAndSpatials() const
        {
            DataLayout l;

            const auto x = X();
            const auto y = Y();
            const auto f = Feature();
            const auto b = Batch();

            DataLayout targetLayout = Tensor::bf;
            switch (layout)
            {
            case Tensor::bf:
            case Tensor::fb:
                return *this;

            case Tensor::fyxb:
                targetLayout = Tensor::fb;

                // TODO: [FUTURE] Use C++17 [[fallthrough]] instead of code duplication to get portable warning avoidance.
                if (f.pitch == y.v*x.v*x.pitch)                                         // no padding in X/Y axis
                {
                    l = targetLayout;
                    break;
                }
                throw std::runtime_error("Unsupported - cannot flatten with padding");

            case Tensor::bfyx:
                if (f.pitch == y.v*x.v*x.pitch)                                         // no padding in X/Y axis
                {
                    l = targetLayout;
                    break;
                }
                throw std::runtime_error("Unsupported - cannot flatten with padding");

            case Tensor::yxfb:
                targetLayout = Tensor::fb;

                // TODO: [FUTURE] Use C++17 [[fallthrough]] instead of code duplication to get portable warning avoidance.
                if ((x.pitch == f.pitch && y.pitch == x.v*x.pitch) ||                   // YX - no Features (val/pitch)
                    (y.v == 1 && x.v == 1 && x.pitch == f.pitch && y.pitch == f.pitch) || // Feature only
                    (f.v * f.pitch == x.pitch && f.v * f.pitch == y.pitch && y.v == 1 && x.v == 1)) // Feature only
                {
                    l = targetLayout;
                    break;
                }
                throw std::runtime_error("Unsupported - cannot flatten yxf to f if f/yx != 1");

            case Tensor::byxf:
                if ((x.pitch == f.pitch && y.pitch == x.v*x.pitch) ||                   // YX - no Features (val/pitch)
                    (y.v == 1 && x.v == 1 && x.pitch == f.pitch && y.pitch == f.pitch) || // Feature only
                    (f.v * f.pitch == x.pitch && f.v * f.pitch == y.pitch && y.v == 1 && x.v == 1)) // Feature only
                {
                    l = targetLayout;
                    break;
                }
                throw std::runtime_error("Unsupported - cannot flatten yxf to f if f/yx != 1");
            default:
                throw std::runtime_error("Unsupported - unsupported layout");
                break;
            }

            DataTensor res = TransformIgnorePadding(l);

            if (l == DataLayout::bf)
            {
                res.dims[Channelndex(l, DataChannelName::BATCH)].pitch = b.pitch;
                res.dims[Channelndex(l, DataChannelName::BATCH)].pad   = b.pad;
            }
            else
            {
                res.dims[Channelndex(l, DataChannelName::FEATURE)].pitch = dims[Channelndex(l, DataChannelName::BATCH) + 1].pitch;
                res.dims[Channelndex(l, DataChannelName::FEATURE)].pad   = dims[Channelndex(l, DataChannelName::BATCH) + 1].pad;
            }

            return res;
        }

        NDims WeightsTensor::GetSimpleDims(const std::vector<size_t>& d, WeightsLayout l)
        {
            std::vector<size_t> newDims = d;

            // TOOD: it's not the right pitches. it's here in order to calculate physical size
            switch (l)
            {
            case os_iyx_osv16:
            case os_iyx_osv16_rotate_180:
                assert(newDims.size() == 4);
                newDims[3] = RoundUp(newDims[3], 16);
                break;
            case os_iyx_osv32:
                assert(newDims.size() == 4);
                newDims[3] = RoundUp(newDims[3], 32);
                break;
            case os_iyx_osv64:
                assert(newDims.size() == 4);
                newDims[3] = RoundUp(newDims[3], 64);
                break;
            case os_i_osv8__ai8:
                assert(newDims.size() == 2);
                newDims[0] = RoundUp(newDims[0], 8);
                newDims[1] = RoundUp(newDims[1], 8);
                break;
            case os_i_osv16__ai8:
                assert(newDims.size() == 2);
                newDims[0] = RoundUp(newDims[0], 8);
                newDims[1] = RoundUp(newDims[1], 16);
                break;
            case os_i_osv16:
                assert(newDims.size() == 2);
                newDims[1] = RoundUp(newDims[1], 16);
                break;
            case i_yxs_os_yxsv2_osv16:
                assert(newDims.size() == 4);
                newDims[0] = RoundUp(newDims[0], 16);
                break;
            case iy_xs_os_xsv2_osv16__ao32:
            case iy_xs_os_xsv2_osv8__ao32:
                assert(newDims.size() == 4);
                newDims[0] = RoundUp(newDims[0], 32);
                break;
            case os_is_yx_isa8_osv8_isv4:
                assert(newDims.size() == 4);
                newDims[3] = RoundUp(newDims[3], 8);
                newDims[2] = RoundUp(newDims[2], 32);
                break;
            case os_is_yx_isa8_osv8_isv4_swizzled_by_4:
                assert(newDims.size() == 4);
                newDims[3] = RoundUp(newDims[3], 32);
                newDims[2] = RoundUp(newDims[2], 32);
                break;
            case is_o_yx_isv32:
                assert(newDims.size() == 4);
                newDims[0] = RoundUp(newDims[0], 32);
                break;
            case is_o32_yx_isv32_swizzled_by_4:
                assert(newDims.size() == 4);
                newDims[0] = RoundUp(newDims[0], 32);
                newDims[3] = RoundUp(newDims[3], 32);
                break;
            case os_is_y_x8_osv8_isv4:
                assert(newDims.size() == 4);
                newDims[2] = RoundUp(newDims[2], 4);
                newDims[3] = RoundUp(newDims[3], 8);
                newDims[0] = RoundUp(newDims[0], 8);
                break;
            case os_is_yx_osv16_isv4:
                assert(newDims.size() == 4);
                newDims[2] = RoundUp(newDims[2], 4);
                newDims[3] = RoundUp(newDims[3], 16);
                break;
            default:
                break;
            }

            NDims ret(newDims.size());
            size_t pitch = 1;

            for (size_t i = 0; i < newDims.size(); i++)
            {
                Pad p = { 0, newDims[i] - d[i] };
                ret[i] = { d[i], pitch, p };
                pitch *= newDims[i];
            }

            if (l == i_yxs_os_yxsv2_osv16)
            {
                ret[3].pitch = RoundUp(ret[1].v * ret[2].v, 2) * ret[1].pitch;
                ret[2].pad.after = newDims[2] - ret[2].v;
            }
            else if (l == iy_xs_os_xsv2_osv16__ao32 ||
                     l == iy_xs_os_xsv2_osv8__ao32)
            {
                ret[2].pitch     = RoundUp(ret[1].v, 2) * ret[1].pitch;
                ret[1].pad.after = newDims[1] - ret[1].v;

                ret[3].pitch     = ret[2].v * ret[2].pitch;
                ret[2].pad.after = newDims[2] - ret[2].v;
            }
            else if (l == os_is_yx_isa8_osv8_isv4 || l == os_is_yx_isa8_osv8_isv4_swizzled_by_4)
            {
                ret[0].pitch = 256;
                ret[1].pitch = ret[0].pitch * ret[0].v;
            }
            else if (l == bf_lyx_yx)
            {
                ret[2].pitch = ret[0].v * ret[1].v * ret[2].v * ret[3].v;
                ret[3].pitch = ret[2].pitch * ret[5].v;
            }

            return ret;
        }

        WeightsTensor WeightsTensor::TransformIgnorePadding(WeightsLayout l, WeightsType t) const
        {
            const uint32_t src_channels = ChannelsCount(layout);
            const uint32_t dst_channels = ChannelsCount(l);

            const size_t src_x = X().v;
            const size_t src_y = Y().v;

            std::vector<size_t> vec(dst_channels);
            if (src_channels == 2 && dst_channels == 2)
            {
                vec[Channelndex(l, WeightsChannelName::IFM)] = IFM().v;
                vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
            }
            else if (src_channels == 4 && dst_channels == 4)
            {
                vec[Channelndex(l, WeightsChannelName::X)] = X().v;
                vec[Channelndex(l, WeightsChannelName::Y)] = Y().v;
                vec[Channelndex(l, WeightsChannelName::IFM)] = IFM().v;
                vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;

                //requirement for winograd 2x3
                if (l == WeightsLayout::winograd_2x3_s1_weights || l == WeightsLayout::winograd_2x3_s1_fused_weights)
                {
                    vec[Channelndex(l, WeightsChannelName::X)] = 4;
                    vec[Channelndex(l, WeightsChannelName::Y)] = 3;
                }
                else if (l == WeightsLayout::winograd_6x3_s1_fused_weights)
                {
                    vec[Channelndex(l, WeightsChannelName::X)] = 8;
                    vec[Channelndex(l, WeightsChannelName::Y)] = 3;
                }
            }
            else if (src_channels == 2 && dst_channels == 4)
            {
                const size_t dst_ifm = IFM().v / (src_x*src_y);
                const size_t dst_xy = IFM().v % (src_x*src_y);
                const size_t dst_y = dst_xy / src_x;
                const size_t dst_x = dst_xy % src_x;
                vec[Channelndex(l, WeightsChannelName::X)] = dst_x;
                vec[Channelndex(l, WeightsChannelName::Y)] = dst_y;
                vec[Channelndex(l, WeightsChannelName::IFM)] = dst_ifm;
                vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
            }
            else if (src_channels == 4 && dst_channels == 2)
            {
                const size_t dst_ifm = IFM().v * src_x * src_y;
                vec[Channelndex(l, WeightsChannelName::IFM)] = dst_ifm;
                vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
            }
            else if (src_channels == 6 && dst_channels == 6)
            {
                vec[Channelndex(l, WeightsChannelName::X)] = IFM().v;
                vec[Channelndex(l, WeightsChannelName::Y)] = OFM().v;
                vec[Channelndex(l, WeightsChannelName::IFM)] = LX().v;
                vec[Channelndex(l, WeightsChannelName::OFM)] = LY().v;
                vec[Channelndex(l, WeightsChannelName::LX)] = X().v;
                vec[Channelndex(l, WeightsChannelName::LY)] = Y().v;
            }
            else
            {
                assert(0);
            }

            return{ vec, t, l };
        }
    }
}