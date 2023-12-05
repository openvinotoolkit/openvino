// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/reshape.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<reshape> : public typed_program_node_base<reshape> {
    using parent = typed_program_node_base<reshape>;
    typed_program_node(const std::shared_ptr<reshape> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    bool has_padding() const {
        return (this->get_output_layout().data_padding || input().get_output_layout(false).data_padding);
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
        if (format::traits(input_layout.format)._order[0] != 0 ||
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

        if (has_padding())
            return false;

        return true;
    }

    void adjust_output_padding() {
        if (!has_padding())
            return;

        auto input_layout = input().get_output_layout(false);
        auto output_layout = this->get_output_layout();
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
