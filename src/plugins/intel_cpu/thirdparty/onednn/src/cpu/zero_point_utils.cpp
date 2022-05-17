/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "cpu/zero_point_utils.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

static void adjust_zero_pad_comp_dims(const dim_t output_dim_size,
        dim_t &estimated_dim_size, dim_t &begin_pad, dim_t &mid_pad,
        dim_t &end_pad) {

    if (output_dim_size < estimated_dim_size) {
        const auto diff = estimated_dim_size - output_dim_size;
        estimated_dim_size = output_dim_size;

        end_pad -= diff;

        if (end_pad < 0) {
            if (mid_pad) {
                mid_pad = 0;
                end_pad += 1;
            }
            if (end_pad < 0) {
                begin_pad += end_pad;
                end_pad = 0;
            }
        }
    }
}

zero_point_pad_comp_config_t::zero_point_pad_comp_config_t(
        const dim_t front_pad, const dim_t back_pad, const dim_t top_pad,
        const dim_t bottom_pad, const dim_t left_pad, const dim_t right_pad,
        const dim_t stride_d, const dim_t stride_h, const dim_t stride_w,
        const dim_t od, const dim_t oh, const dim_t ow)

    : top_pad(utils::div_up(top_pad, stride_h))
    , bottom_pad(utils::div_up(bottom_pad, stride_h))
    , left_pad(utils::div_up(left_pad, stride_w))
    , right_pad(utils::div_up(right_pad, stride_w))
    , front_pad(utils::div_up(front_pad, stride_d))
    , back_pad(utils::div_up(back_pad, stride_d))
    , mid_h((oh - this->top_pad - this->bottom_pad > 0)
                              && (this->left_pad > 0 || this->right_pad > 0
                                      || this->front_pad > 0 || this->back_pad)
                      ? 1
                      : 0)
    , mid_w((ow - this->left_pad - this->right_pad > 0)
                              && (this->bottom_pad > 0 || this->top_pad > 0
                                      || this->front_pad > 0 || this->back_pad)
                      ? 1
                      : 0)
    , mid_d((od - this->front_pad - this->back_pad > 0)
                              && (this->top_pad > 0 || this->bottom_pad > 0
                                      || this->right_pad > 0 || this->left_pad)
                      ? 1
                      : 0)
    , h(this->top_pad + this->bottom_pad + this->mid_h)
    , w(this->left_pad + this->right_pad + this->mid_w)
    , d(this->front_pad + this->back_pad + this->mid_d) {

    adjust_zero_pad_comp_dims(
            oh, this->h, this->top_pad, this->mid_h, this->bottom_pad);
    adjust_zero_pad_comp_dims(
            ow, this->w, this->left_pad, this->mid_w, this->right_pad);
    adjust_zero_pad_comp_dims(
            od, this->d, this->front_pad, this->mid_d, this->back_pad);
}

zero_point_config_t::zero_point_config_t(const primitive_attr_t &attr)
    : src_exists(!attr.zero_points_.has_default_values(DNNL_ARG_SRC))
    , dst_exists(!attr.zero_points_.has_default_values(DNNL_ARG_DST))
    , src_is_common(attr.zero_points_.common(DNNL_ARG_SRC)) {}

bool zero_point_config_t::zp_exists() const noexcept {
    return src_exists || dst_exists;
}

zero_point_call_params_t::zero_point_call_params_t(const int32_t *src,
        const int32_t *dst, const int32_t *src_comp,
        const int32_t *src_pad_comp)
    : src(src), dst(dst), src_comp(src_comp), src_pad_comp(src_pad_comp) {}

bool zero_points_valid(
        const primitive_attr_t *attr, bool per_oc_bcast_accepted) noexcept {

    int mask_src = -1, mask_dst = -1;
    static constexpr int c_mask = 0x1,
                         g_mask = 0x3; // mask for i/o-channel and ngroups

    attr->zero_points_.get(DNNL_ARG_SRC, nullptr, &mask_src, nullptr);
    attr->zero_points_.get(DNNL_ARG_DST, nullptr, &mask_dst, nullptr);

    const bool src_mask_valid = per_oc_bcast_accepted
            ? utils::one_of(mask_src, 0, c_mask, g_mask)
            : mask_src == 0;
    const bool dst_mask_valid = per_oc_bcast_accepted
            ? utils::one_of(mask_dst, 0, c_mask, g_mask)
            : mask_dst == 0;

    return attr->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
            && src_mask_valid && dst_mask_valid;
}

void set_zp_src_comp_flags(memory_desc_t &weights_md, bool with_groups) {
    weights_md.extra.flags
            |= memory_extra_flags::compensation_conv_asymmetric_src;
    weights_md.extra.asymm_compensation_mask
            = (1 << 0) + (with_groups ? (1 << 1) : 0);
}

const int32_t *get_src_zp_comp_from_wei(const int8_t *weights,
        const memory_desc_wrapper &weights_md, bool signed_input, dim_t ngroups,
        dim_t oc) {

    const auto comp_offset
            = weights_md.size() - weights_md.additional_buffer_size();
    const auto src_zp_com_offset = signed_input ? ngroups * oc : 0;
    return reinterpret_cast<const int32_t *>(&weights[comp_offset])
            + src_zp_com_offset;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
