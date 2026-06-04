// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "openvino/core/partial_shape.hpp"
#include "crop_inst.h"
#include "mvn_inst.h"
#include "vl_sdpa_inst.h"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

padding propagate_padding(const layout& in_layout, const ov::PartialShape& out_shape, reshape::reshape_mode mode, const ov::ITensorAccessor& ta);

template <>
struct typed_program_node<reshape> : public typed_program_node_base<reshape> {
    using parent = typed_program_node_base<reshape>;
    typed_program_node(const std::shared_ptr<reshape> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
        set_runtime_skippable(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    bool is_runtime_propagatable_padding() const {
        auto prim = typed_desc();
        if (prim->mode == reshape::reshape_mode::squeeze || prim->mode == reshape::reshape_mode::unsqueeze) {
            // For proper padding propagation we need to know output pattern at model loading stage
            // in case of squeeze/unsqueeze mode
            if (prim->output_pattern.empty())
                return false;

            if (input().is_type<crop>() && prim->mode == reshape::reshape_mode::squeeze) {
                const auto crop_axis = input().as<crop>().get_primitive()->axis;
                const auto& output_pattern = prim->output_pattern;

                // Do not propagate output padding in squeeze mode if the squeezed dimension corresponds to the crop axis
                return std::find(output_pattern.begin(), output_pattern.end(), crop_axis) == output_pattern.end();
            }

            return true;
        }

        // TODO: This function is to limit condition to a specific case (crop + reshape) among cases for the base mode
        if (!input().is_type<crop>())
            return false;

        // oneDNN supports padded input of outer axis only for buffer fusing on static shape
        if (!has_outer_padding_offset() && get_users().size() == 1 && get_users().front()->get_preferred_impl_type() == impl_types::onednn)
            return false;

        // MVN canonicalizes input strides and cannot tolerate dynamic padding offsets.
        if (get_users().size() == 1 && get_users().front()->is_type<mvn>())
            return false;

        // vl_sdpa uses a CM kernel that receives raw SVM pointers.  Generic dynamic
        // padding (e.g. from an inner-axis crop) cannot be applied through shape_info
        // as with OCL kernels.  The one exception is the TransposeSplitMatcher axis=1
        // pattern: crop axis=1 with a size-1 input[1] followed by a rank-reducing reshape.
        // In that case the CM kernel receives dedicated token_offset_q / token_offset_kv
        // scalars computed from _lower_size[1], so propagation IS safe.
        // All other vl_sdpa paths remain blocked.
        if (get_users().size() == 1 && get_users().front()->is_type<vl_sdpa>()) {
            // Allow only if the crop is the axis=1 / size-1 squeeze pattern handled by
            // the token_offset scalars in the CM kernel.  That pattern is already checked
            // below (axis == 1 block), so we fall through here without blocking.
            auto axis = input().as<crop>().get_primitive()->axis;
            const auto& input_pshape = input().get_output_layout(false).get_partial_shape();
            const bool is_axis1_size1_squeeze =
                axis == 1 &&
                !input_pshape[1].is_dynamic() &&
                input_pshape[1].get_length() == 1 &&
                prim->output_partial_shape.size() + 1 == input_pshape.size();
            if (!is_axis1_size1_squeeze)
                return false;
        }

        auto axis = input().as<crop>().get_primitive()->axis;
        const auto& input_pshape = input().get_output_layout(false).get_partial_shape();

        // directly to the reshape's leading axis when the cropped batch size is exactly 1.
        // Keep the same safeguard as for inner-axis crops: if the reshape output pattern is
        // not known at model loading time, do not enable runtime padding propagation.
        if (axis == 0 && !input_pshape[0].is_dynamic()) {
            if (prim->output_pattern.empty())
                return false;
            if (input_pshape[0].get_length() != 1)
                return false;
            // Reject if the reshape preserves the batch=1 dim (spatial flatten, not batch squeeze).
            // output_pattern[0] == -1 means the first dim is inferred (batch absorbed/squeezed).
            // output_pattern[0] == 0 or 1 means batch=1 is explicitly kept.
            // e.g. [1,C,H,W] -> [1,C,H*W] has pattern [1,-1] => reject
            //      [1,N,H,W,C] -> [N,H,W,C] has pattern [-1,H,W,C] => allow
            auto first_out_pattern = prim->output_pattern[0];
            if (first_out_pattern == 0 || first_out_pattern == 1)
                return false;
            return true;
        }

        // TransposeSplitMatcher optimization: when Transpose+Split(axis=0) over a
        // [-1, 3, H, S] QKV tensor is replaced by Split(axis=1), each crop output
        // has shape [-1, 1, H, S] with axis=1.  The reshape that follows squeezes
        // the size-1 dim-1, producing [-1, H, S].
        //
        // Why this pattern is safe to propagate padding through:
        //   - The crop offset lives on axis=1 (the "3" dimension, i.e. Q/K/V slot).
        //   - The reshape only removes that size-1 dimension; it does NOT reorder or
        //     merge any bytes.  The memory layout of the remaining dimensions
        //     (batch=-1, H, S) is therefore identical before and after the reshape,
        //     and the buffer pointer + pitch of the outer dynamic batch dimension is
        //     still valid after the squeeze.
        //   - The crop size along axis=1 is always exactly 1 (static), so the
        //     offset is a known, constant stride-multiple — no runtime-shape-info
        //     lookup is required to compute it.
        //   - Downstream consumers (RoPE, oneDNN SDPA) only receive the squeezed
        //     [-1, H, S] view and never observe the axis=1 padding, so no consumer
        //     needs to be updated.
        //
        // Conditions checked:
        //   1. axis == 1                      (crop is on the QKV-slot dimension)
        //   2. input_pshape[1] is static 1    (exactly one Q/K/V slice per crop)
        //   3. output rank == input rank - 1   (reshape only drops that size-1 dim)
        //
        // Note: output_pattern is NOT required here.  The safety argument depends
        // only on the input crop axis and the output rank, not on the specific
        // output_pattern values.  In practice the Qwen3-VL model uses a dynamic
        // shape tensor (not a compile-time constant) for the Reshape second input,
        // leaving output_pattern empty — requiring it would incorrectly block this
        // case even though conditions 1–3 are fully sufficient.
        if (axis == 1 && !input_pshape[1].is_dynamic() && input_pshape[1].get_length() == 1) {
            if (prim->output_partial_shape.size() + 1 == input_pshape.size()) {
                return true;
            }
        }

        auto input_rank = input_pshape.size();
        auto input_last_dim = static_cast<int64_t>(input_rank - 1);
        if (axis != input_last_dim || input_pshape[input_last_dim].is_dynamic())
            return false;

        auto input_last_dim_val = input_pshape[input_last_dim].get_length();
        const auto& output_pshape = prim->output_partial_shape;
        // TODO: If the reshape's output shape is non constant, issue occurs
        // during shape inference due to execution order at runtime
        if (prim->output_pattern.empty())
            return false;

        // Iteratively check the total product of all static innermost dimensions
        // until the crop dimension value matches or the first dynamic dimension is encountered
        int64_t mul = 1;
        size_t matched_trailing_dims = 0;
        for (size_t i = output_pshape.size(); i > 1 ; i--) {
            if (output_pshape[i - 1].is_dynamic() || mul == input_last_dim_val)
                break;

            mul *= output_pshape[i - 1].get_length();
            matched_trailing_dims++;
        }
        if (input_last_dim_val != mul)
            return false;
        // Reject when reshape drops the cropped axis (e.g. [N,M,1] -> [N,M]): no output axis can
        // carry the cropped axis padding, so sibling crop outputs would all point to the same
        // base buffer region (aliased).
        if (matched_trailing_dims == 0 && output_pshape.size() < input_pshape.size())
            return false;

        return true;
    }

    bool has_padding() const {
        return (this->get_output_layout().data_padding
                || input().get_output_layout(false).data_padding
                || input().get_output_layout(false).data_padding.is_dynamic());
    }

    bool has_outer_padding_offset() const {
        if (!has_padding())
            return false;

        auto input_layout = input().get_output_layout(false);
        auto input_pad = input_layout.data_padding;
        for (size_t i = 0 ; i < input_layout.get_spatial_rank() ; i++) {
            if (input_pad._lower_size[2 + i] != 0)
                return false;
            if (input_pad._upper_size[2 + i] != 0)
                return false;
        }

        // Expected a padded input of only batch axis with 'bxxx' format
        if (input_layout.format.dims_order()[0] != 0 ||
            input_pad._lower_size[1] != 0 ||
            input_pad._upper_size[1] != 0)
            return false;

        if (format::is_multi_blocked(input_layout.format))
            return false;

        // Outer padding exists. It might need to update padding size of output layout
        return true;
    }

    bool is_in_place() const {
        if (this->is_output() || this->has_fused_primitives())
            return false;

        if (input().get_output_layout(false).data_padding.is_dynamic() && is_runtime_propagatable_padding())
            return true;

        if (has_padding())
            return false;

        return true;
    }

    void adjust_output_padding() {
        if (!has_padding())
            return;

        auto input_layout = input().get_output_layout(false);
        auto output_layout = this->get_output_layout();
        if (input_layout.data_padding.is_dynamic()) {
            auto prim = typed_desc();
            // TODO: If outer padding exists, ouput padding propagation is not supported in the base mode
            if (prim->mode == reshape::reshape_mode::base)
                return;

            ov::PartialShape pattern_shape = { static_cast<int64_t>(prim->output_pattern.size()) };
            if (pattern_shape.size() == 0)
                pattern_shape = {};

            auto pattern_data = prim->output_pattern;
            auto pattern_tensor = make_tensor({pattern_shape, data_types::i64, format::bfyx}, static_cast<void*>(pattern_data.data()));
            std::unordered_map<size_t, ov::Tensor> const_data {{1, pattern_tensor}};
            const auto ta = ov::make_tensor_accessor(const_data);

            this->set_output_padding(propagate_padding(input_layout, output_layout.get_partial_shape(), prim->mode, ta));
        } else {
            // adjust output padding if Reshape has an outer padding exists in an input
            auto input_pitches = input_layout.get_pitches();
            auto input_pad = input_layout.data_padding;
            size_t first_element_offset = input_pad._lower_size[0];
            // feature and spatial size
            first_element_offset *= input_pitches[0];

            size_t inner_size = 1;
            for (size_t i = 0 ; i < output_layout.get_spatial_rank() ; i++) {
                inner_size *= output_layout.spatial(i);
            }
            inner_size *= output_layout.feature();

            auto new_batch_pad = first_element_offset / inner_size;
            this->set_output_padding(cldnn::padding({static_cast<int32_t>(new_batch_pad), 0, 0, 0}, {0, 0, 0, 0}));
        }

        return;
    }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {1}; }
};

using reshape_node = typed_program_node<reshape>;

template <>
class typed_primitive_inst<reshape> : public typed_primitive_inst_base<reshape> {
    using parent = typed_primitive_inst_base<reshape>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(reshape_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(reshape_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(reshape_node const& node);

    typed_primitive_inst(network& network, reshape_node const& node);

    void update_output_memory() override;

private:
    void on_execute() override;
};

using reshape_inst = typed_primitive_inst<reshape>;

}  // namespace cldnn
