/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_JIT_CONV_POST_OP_SUPPORT_HPP
#define GPU_JIT_CONV_POST_OP_SUPPORT_HPP

#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/eltwise_pd.hpp"
#include "common/primitive_attr.hpp"
#include "common/utils.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/kernel_arg_info.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class eltwise_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(eltwise_t, func_impl_t)

    static func_t make(
            alg_kind_t alg_kind, float scale, float alpha, float beta) {
        return func_t(new eltwise_t(alg_kind, scale, alpha, beta));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (alg_kind == other.alg_kind) && (scale == other.scale)
                && (alpha == other.alpha) && (beta == other.beta);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(alg_kind, scale, alpha, beta);
    }

    std::string str() const override {
        switch (alg_kind) {
            case alg_kind::eltwise_relu: return "relu";
            case alg_kind::eltwise_tanh: return "tanh";
            case alg_kind::eltwise_elu: return "elu";
            case alg_kind::eltwise_square: return "square";
            case alg_kind::eltwise_abs: return "abs";
            case alg_kind::eltwise_sqrt: return "sqrt";
            case alg_kind::eltwise_swish: return "swish";
            case alg_kind::eltwise_linear: return "linear";
            case alg_kind::eltwise_bounded_relu: return "bounded_relu";
            case alg_kind::eltwise_soft_relu: return "soft_relu";
            case alg_kind::eltwise_logistic: return "logistic";
            case alg_kind::eltwise_logsigmoid: return "logsigmoid";
            case alg_kind::eltwise_mish: return "mish";
            case alg_kind::eltwise_exp: return "exp";
            case alg_kind::eltwise_log: return "log";
            case alg_kind::eltwise_clip: return "clip";
            case alg_kind::eltwise_clip_v2: return "clip_v2";
            case alg_kind::eltwise_pow: return "pow";
            case alg_kind::eltwise_gelu_tanh: return "gelu_tanh";
            case alg_kind::eltwise_gelu_erf: return "gelu_erf";
            case alg_kind::eltwise_hardswish: return "hardswish";
            case alg_kind::eltwise_relu_use_dst_for_bwd:
                return "relu_use_dst_for_bwd";
            case alg_kind::eltwise_tanh_use_dst_for_bwd:
                return "tanh_use_dst_for_bwd";
            case alg_kind::eltwise_elu_use_dst_for_bwd:
                return "elu_use_dst_for_bwd";
            case alg_kind::eltwise_sqrt_use_dst_for_bwd:
                return "sqrt_use_dst_for_bwd";
            case alg_kind::eltwise_logistic_use_dst_for_bwd:
                return "logistic_use_dst_for_bwd";
            case alg_kind::eltwise_exp_use_dst_for_bwd:
                return "exp_use_dst_for_bwd";
            case alg_kind::eltwise_clip_v2_use_dst_for_bwd:
                return "clip_v2_use_dst_for_bwd";
            case alg_kind::eltwise_round: return "round";
            default: ir_error_not_expected();
        }
        return "unknown";
    }

    IR_DEFINE_ARG_GET(elems, 0)
    IR_DEFINE_ARG_GET(data, 1)

    alg_kind_t alg_kind;
    float scale;
    float alpha;
    float beta;

private:
    eltwise_t(alg_kind_t alg_kind, float scale, float alpha, float beta)
        : alg_kind(alg_kind), scale(scale), alpha(alpha), beta(beta) {}
};

class post_op_t {
public:
    post_op_t() = default;

    post_op_t(const view_t &rhs_view, const expr_t &rhs_buf, uint32_t rhs_mask,
            float rhs_scale, op_kind_t op_kind)
        : rhs_view_(rhs_view)
        , rhs_buf_(rhs_buf)
        , rhs_mask_(rhs_mask)
        , rhs_scale_(rhs_scale)
        , op_kind_(op_kind) {}

    post_op_t(const func_t &eltwise) : eltwise_(eltwise) {}

    const view_t &rhs_view() const { return rhs_view_; }

    const expr_t &rhs_buf() const { return rhs_buf_; }

    uint32_t rhs_mask() const { return rhs_mask_; }

    float rhs_scale() const { return rhs_scale_; }

    op_kind_t op_kind() const { return op_kind_; }

    const func_t &eltwise() const { return eltwise_; }

    bool has_rhs() const { return !rhs_view_.is_empty(); }

    bool needs_load() const {
        if (!has_rhs()) return false;
        if (!rhs_buf_.type().is_ptr()) return false;
        return true;
    }

    bool is_broadcast_dim(int dim_idx) const {
        return (rhs_mask() & (1 << dim_idx)) == 0;
    }

    tensor_t apply_mask(const tensor_t &tile) const {
        ir_assert(has_rhs());
        ir_assert(tile.ndims() == rhs_view_.nvdims())
                << "Incompatible dimensions.";
        auto tile_dims = apply_mask(tile.dims(), 1);
        auto tile_start = apply_mask(tile.start(), 0);
        return tensor_t(tile_dims, tile_start);
    }

    template <typename T, typename U>
    std::vector<T> apply_mask(
            const std::vector<T> &v, const U &mask0_value) const {
        ir_assert(has_rhs());
        ir_assert(int(v.size()) == rhs_view_.nvdims())
                << "Incompatible dimensions.";
        auto ret = v;
        for (int i = 0; i < int(v.size()); i++) {
            if ((rhs_mask_ & (1 << i)) == 0) { ret[i] = mask0_value; }
        }
        return ret;
    }

    post_op_t create_sub_post_op(const tensor_t &tile) const {
        if (!has_rhs()) return *this;

        auto ret = *this;
        ret.rhs_view_ = ret.rhs_view_.create_sub_view(apply_mask(tile));
        return ret;
    }

private:
    view_t rhs_view_;
    expr_t rhs_buf_;
    uint32_t rhs_mask_ = 0;
    float rhs_scale_ = 1;
    op_kind_t op_kind_ = op_kind_t::undef;
    func_t eltwise_;
};

inline op_kind_t alg_kind_to_op_kind(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::binary_add: return op_kind_t::_add;
        case alg_kind::binary_sub: return op_kind_t::_sub;
        case alg_kind::binary_mul: return op_kind_t::_mul;
        case alg_kind::binary_div: return op_kind_t::_div;
        case alg_kind::binary_min: return op_kind_t::_min;
        case alg_kind::binary_max: return op_kind_t::_max;
        case alg_kind::binary_ge: return op_kind_t::_ge;
        case alg_kind::binary_gt: return op_kind_t::_gt;
        case alg_kind::binary_le: return op_kind_t::_le;
        case alg_kind::binary_lt: return op_kind_t::_lt;
        case alg_kind::binary_eq: return op_kind_t::_eq;
        case alg_kind::binary_ne: return op_kind_t::_ne;
        default: ir_error_not_expected();
    }
    return op_kind_t::undef;
}

class post_op_context_t {
public:
    post_op_context_t() = default;

    post_op_context_t(const convolution_pd_t *pd, const conv_config_t &cfg,
            const view_t &lhs_view, const kernel_arg_info_t &kernel_arg_info)
        : pd_(pd), cfg_(&cfg), lhs_view_(lhs_view) {

        // Handle bias.
        if ((pd->is_fwd() || pd->is_bwd_d()) && pd->with_bias()) {
            uint32_t rhs_mask = convert_rhs_mask(2); // Per-channel mask.
            auto rhs_view = create_rhs_view(
                    pd->invariant_bia_md()->data_type, rhs_mask);
            auto rhs_buf = kernel_arg_info.find_arg("bia");
            post_ops_.emplace_back(
                    rhs_view, rhs_buf, rhs_mask, 1, op_kind_t::_add);
        }

        auto *attr = pd->attr();

        // Handle output scales.
        bool with_oscales = !attr->output_scales_.has_default_values();
        if (with_oscales) {
            uint32_t mask = convert_rhs_mask(attr->output_scales_.mask_);
            auto oscales_buf = kernel_arg_info.find_arg("oscales");
            auto oscales_view = create_rhs_view(type_t::f32(), mask);
            post_ops_.emplace_back(
                    oscales_view, oscales_buf, mask, 1, op_kind_t::_mul);
        }

        // Handle post-ops.
        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_eltwise()) {
                post_ops_.emplace_back(eltwise_t::make(po.eltwise.alg,
                        po.eltwise.scale, po.eltwise.alpha, po.eltwise.beta));
            } else if (po.is_sum(/*require_scale_one=*/false)) {
                float scale = po.sum.scale;
                uint32_t rhs_mask = 0xFFFFFFFF;
                auto rhs_buf = kernel_arg_info.find_arg(
                        pd->is_fwd() ? "dst" : "src");
                auto rhs_view = lhs_view_;
                if (po.sum.dt != data_type::undef)
                    rhs_view = rhs_view.retype(po.sum.dt);
                post_ops_.emplace_back(
                        rhs_view, rhs_buf, rhs_mask, scale, op_kind_t::_add);
            } else if (po.is_binary()) {
                uint32_t rhs_mask = 0;
                auto rhs_view = create_rhs_view(po.binary.src1_desc, rhs_mask);
                auto buf_name = "binary_rhs_" + std::to_string(i);
                auto rhs_buf = kernel_arg_info.find_arg(buf_name);
                post_ops_.emplace_back(rhs_view, rhs_buf, rhs_mask, 1,
                        alg_kind_to_op_kind(po.binary.alg));
            } else {
                ir_error_not_expected();
            }
        }
    }

    const std::vector<post_op_t> &post_ops() const { return post_ops_; }

    int lhs_ndims() const { return lhs_view_.nvdims(); }

    dim_t lhs_dim(int idx) const { return lhs_view_.vdims()[idx]; }

    dim_t lhs_padded_dim(int idx) const { return lhs_view_.tlayout().dim(idx); }

    bool has_lhs_mask(int idx) const { return lhs_view_.has_tmask(idx); }

    bool is_lhs_dim_zero_padded(int idx) const {
        if (has_lhs_mask(idx)) return true;
        if (lhs_dim(idx) != lhs_padded_dim(idx)) return true;
        return false;
    }

    bool need_to_restore_zero_padding() const {
        auto *attr = pd_->attr();
        if (pd_->with_bias() && pd_->is_fwd()
                && pd_->dst_md()->dims[0] != pd_->dst_md()->padded_dims[0])
            return true;
        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_eltwise()) {
                if (!eltwise_fwd_pd_t::eltwise_preserves_zero(po.eltwise))
                    return true;
            } else if (po.is_sum(/*require_scale_one=*/false)) {
                // Preserves zero padding.
            } else if (po.is_binary()) {
                for (int j = 0; j < lhs_ndims(); j++) {
                    if (!is_lhs_dim_zero_padded(j)) continue;
                    // Check if binary preserves zeros: (0 op X == 0) or (0 op 0 == 0).
                    bool zero_op_x_ok = utils::one_of(po.binary.alg,
                            alg_kind::binary_mul, alg_kind::binary_div);
                    bool zero_op_zero_ok = zero_op_x_ok
                            || utils::one_of(po.binary.alg,
                                    alg_kind::binary_add, alg_kind::binary_sub,
                                    alg_kind::binary_min, alg_kind::binary_max,
                                    alg_kind::binary_gt, alg_kind::binary_lt,
                                    alg_kind::binary_ne);

                    uint32_t rhs_mask
                            = utils::get_dims_mask(lhs_view_.vdims().data(),
                                    po.binary.src1_desc.dims, lhs_ndims());
                    if ((rhs_mask & (1 << j)) == 0 && !zero_op_x_ok)
                        return true;
                    if (!zero_op_zero_ok) return true;
                }
            } else {
                ir_error_not_expected();
            }
        }
        return false;
    }

private:
    // rhs tensor has plain layout.
    view_t create_rhs_view(const type_t &type, uint32_t rhs_mask) const {
        std::vector<dim_t> rhs_dims = lhs_view_.vdims();
        uint32_t bound_check_mask = 0;
        for (int i = 0; i < lhs_ndims(); i++) {
            if ((rhs_mask & (1 << i)) == 0) {
                // Broadcast dimension.
                rhs_dims[i] = 1;
            } else if (lhs_padded_dim(i) != lhs_dim(i)) {
                bound_check_mask |= (1 << i);
            } else if (has_lhs_mask(i)) {
                bound_check_mask |= (1 << i);
            }
        }
        return view_t(layout_t(type, 0, rhs_dims, /*do_normalize=*/false),
                lhs_view_.vvars(), bound_check_mask);
    }

    // rhs tensor layout is defined by rhs_md.
    view_t create_rhs_view(
            const memory_desc_t &rhs_md, uint32_t &rhs_mask) const {
        bool add_groups
                = (lhs_ndims() == 6); // Add groups to match ngcdhw layout.
        layout_t rhs_layout(rhs_md, /*do_normalize=*/false);
        std::vector<dim_t> rhs_dims(rhs_md.dims, rhs_md.dims + rhs_md.ndims);
        std::vector<dim_t> rhs_padded_dims(
                rhs_md.padded_dims, rhs_md.padded_dims + rhs_md.ndims);
        maybe_reshape_rhs_dims(
                cfg_->ndims, rhs_layout, rhs_dims, rhs_padded_dims);
        rhs_layout = normalize_conv_layout(rhs_layout, /*with_groups=*/false,
                cfg_->g, cfg_->is_dw, cfg_->reduced_to_1d, add_groups,
                /*is_wei=*/false);
        rhs_dims = normalize_conv_dims(rhs_dims, /*with_groups=*/false, cfg_->g,
                cfg_->is_dw, cfg_->reduced_to_1d, add_groups, /*is_wei=*/false);
        rhs_padded_dims = normalize_conv_dims(rhs_padded_dims,
                /*with_groups=*/false, cfg_->g, cfg_->is_dw,
                cfg_->reduced_to_1d, add_groups, /*is_wei=*/false);
        ir_assert(rhs_layout.ndims() == lhs_ndims())
                << "Incompatible dimensions.";
        uint32_t bound_check_mask = 0;
        for (int i = 0; i < lhs_ndims(); i++) {
            if (rhs_dims[i] == 1) continue; // Broadcast, no bound check needed.
            if (rhs_padded_dims[i] != lhs_padded_dim(i)) {
                bound_check_mask |= (1 << i);
            } else if (has_lhs_mask(i)) {
                bound_check_mask |= (1 << i);
            }
        }
        rhs_mask = utils::get_dims_mask(
                lhs_view_.vdims().data(), rhs_dims.data(), lhs_ndims());
        return view_t(rhs_layout, lhs_view_.vvars(), bound_check_mask);
    }

    static void maybe_reshape_rhs_dims(int ndims, layout_t &rhs_layout,
            std::vector<dim_t> &rhs_dims, std::vector<dim_t> &rhs_padded_dims) {
        ir_assert(rhs_layout.ndims() == int(rhs_dims.size()));
        if (rhs_layout.ndims() < ndims) {
            rhs_layout = layout_t(rhs_layout.type(), ndims, rhs_layout.offset(),
                    rhs_layout.blocks(), /*do_normalize=*/false);
            rhs_dims.resize(ndims, 1);
            rhs_padded_dims.resize(ndims, 1);
        }
    }

    uint32_t convert_rhs_mask(uint32_t orig_mask) const {
        bool add_groups
                = (lhs_ndims() == 6); // Add groups to match ngcdhw layout.
        int orig_ndims
                = 2 + cfg_->ndims; // number of dimensions before normalization.
        std::vector<dim_t> dummy_dims(orig_ndims, 1);
        dim_t mask_set_value = 2;
        for (int i = 0; i < orig_ndims; i++) {
            if ((orig_mask & (1 << i)) != 0) dummy_dims[i] = mask_set_value;
        }
        auto cvt_dims = normalize_conv_dims(dummy_dims, /*with_groups=*/false,
                cfg_->g, cfg_->is_dw, cfg_->reduced_to_1d, /*add_groups=*/false,
                /*is_wei=*/false);
        // Split channels into groups and channels to match ngcdhw layout.
        if (add_groups) cvt_dims.insert(cvt_dims.begin() + 1, cvt_dims[1]);
        ir_assert(int(cvt_dims.size()) == lhs_ndims());

        uint32_t mask = 0;
        for (int i = 0; i < lhs_ndims(); i++) {
            if (cvt_dims[i] == mask_set_value) mask = mask | (1 << i);
        }
        return mask;
    }

    const convolution_pd_t *pd_;
    const conv_config_t *cfg_;
    view_t lhs_view_;
    std::vector<post_op_t> post_ops_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
