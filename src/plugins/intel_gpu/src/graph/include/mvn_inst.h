// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/mvn.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<mvn> : public typed_program_node_base<mvn> {
    using parent = typed_program_node_base<mvn>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using mvn_node = typed_program_node<mvn>;

template <>
class typed_primitive_inst<mvn> : public typed_primitive_inst_base<mvn> {
    using parent = typed_primitive_inst_base<mvn>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(mvn_node const& /*node*/, const kernel_impl_params& impl_param)  {
        auto in_layout = impl_param.get_input_layout(0);
        auto output_type = impl_param.desc->output_data_types[0].value_or(in_layout.data_type);

        // Normalized output is fractional: an i8/u8 output type is only valid via a fused quantize.
        if (impl_param.has_fused_primitives()) {
            output_type = impl_param.get_output_element_type();
        } else if (data_type_traits::is_i8_u8(in_layout.data_type)) {
            output_type = data_types::f32;
        }

        return {layout(in_layout.get<ShapeType>(), output_type, in_layout.format)};
    }
    static layout calc_output_layout(mvn_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(mvn_node const& node);

    typed_primitive_inst(network& network, mvn_node const& node);
};

using mvn_inst = typed_primitive_inst<mvn>;

}  // namespace cldnn
