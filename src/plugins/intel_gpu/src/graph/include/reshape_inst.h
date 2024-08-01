// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "openvino/core/partial_shape.hpp"
#include "crop_inst.h"
#include "rope_inst.h"
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
        if (prim->mode == reshape::reshape_mode::squeeze || prim->mode == reshape::reshape_mode::unsqueeze)
            return true;

        // TODO: This function is to limit condition to a specific case (crop + reshape) among cases for the base mode
        if (!input().is_type<crop>())
            return false;

        // TODO: If user is RoPE and dynamic padding exists, ouput padding propagation is not supported in the base mode
        if (get_users().size() == 1 && get_users().front()->is_type<rope>())
            return false;

        auto axis = input().as<crop>().get_primitive()->axis;
        const auto& input_pshape = input().get_output_layout(false).get_partial_shape();
        auto input_rank = input_pshape.size();
        auto input_last_dim = static_cast<int64_t>(input_rank - 1);
        if (axis != input_last_dim || input_pshape[input_last_dim].is_dynamic())
            return false;

        auto input_last_dim_val = input_pshape[input_last_dim].get_length();
        const auto& output_pshape = prim->output_partial_shape;
        // TODO: If the reshape's output shape is non constant, issue occurs
        // during shape inference due to execution order at runtime
        if ((output_pshape.size() != input_rank + 1) || prim->output_pattern.empty())
            return false;

        int64_t mul = 1;
        for (size_t i = input_rank - 1; i < output_pshape.size() ; i++) {
            if (output_pshape[i].is_dynamic())
                return false;
            mul *= output_pshape[i].get_length();
        }
        if (input_last_dim_val != mul)
            return false;

        return true;
    }

    bool has_padding() const {
        return (this->get_output_layout().data_padding || input().get_output_layout(false).data_padding || input().get_output_layout(false).has_dynamic_pad());
    }

    bool has_outer_padding_offset() const {
        if (!has_padding())
            return false;

        auto input_layout = input().get_output_layout(false);
        auto input_pad = input_layout.data_padding;
        for (size_t i = 0 ; i < input_layout.get_spatial_rank() ; i++) {
            if (input_pad.lower_size().spatial[i] != 0)
                return false;
            if (input_pad.upper_size().spatial[i] != 0)
                return false;
        }

        // Expected a padded input of only batch axis with 'bxxx' format
        if (input_layout.format.dims_order()[0] != 0 ||
            input_pad.lower_size().feature[0] != 0)
            return false;

        if (format::is_multi_blocked(input_layout.format))
            return false;

        // Outer padding exists. It might need to update padding size of output layout
        return true;
    }

    bool is_in_place() const {
        if (this->is_output() || this->has_fused_primitives())
            return false;

        if (input().get_output_layout(false).has_dynamic_pad() && is_runtime_propagatable_padding())
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
        if (input_layout.has_dynamic_pad()) {
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
            return;
        }

        if (input_layout.batch() != output_layout.batch()) {
            // adjust output padding if Reshape has an outer padding exists in an input
            auto input_pitches = input_layout.get_pitches();
            auto input_pad = input_layout.data_padding;
            size_t first_element_offset = input_pad.lower_size().batch[0];
            // feature and spatial size
            first_element_offset *= input_pitches.batch[0];

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
