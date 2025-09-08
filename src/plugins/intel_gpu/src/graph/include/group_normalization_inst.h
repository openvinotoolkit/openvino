// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "intel_gpu/primitives/group_normalization.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<group_normalization> : public typed_program_node_base<group_normalization> {
    using parent = typed_program_node_base<group_normalization>;

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using group_normalization_node = typed_program_node<group_normalization>;

template <>
class typed_primitive_inst<group_normalization> : public typed_primitive_inst_base<group_normalization> {
    using parent = typed_primitive_inst_base<group_normalization>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(group_normalization_node const& /*node*/, const kernel_impl_params& impl_param) {
        auto in_layout = impl_param.get_input_layout(0);
        auto out_format = in_layout.format;
        auto output_type = impl_param.desc->output_data_types[0].value_or(in_layout.data_type);

        if (impl_param.has_fused_primitives()) {
            output_type = impl_param.get_output_element_type();
            for (auto& desc : impl_param.fused_desc) {
                if (desc.is_type<reorder>()) {
                    out_format = desc.output_layout.format;
                }
            }
        }

        return { layout(in_layout.get<ShapeType>(), output_type, out_format) };
    }

    static layout calc_output_layout(group_normalization_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(group_normalization_node const& node);

    typed_primitive_inst(network& network, group_normalization_node const& desc);
};

using group_normalization_inst = typed_primitive_inst<group_normalization>;

}  // namespace cldnn
