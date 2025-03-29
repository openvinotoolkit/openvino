// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using custom_gpu_primitive_node = typed_program_node<custom_gpu_primitive>;

template <>
class typed_primitive_inst<custom_gpu_primitive> : public typed_primitive_inst_base<custom_gpu_primitive> {
    using parent = typed_primitive_inst_base<custom_gpu_primitive>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(custom_gpu_primitive_node const& /*node*/, const kernel_impl_params& impl_param) {
        assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
               "Output data type forcing is not supported for "
               "custom_gpu_primitive_node!");
        layout output_layout = impl_param.typed_desc<custom_gpu_primitive>()->output_layout;

        // if the output layout format was set to any, it means the layer output format will be the same as the first input
        if (output_layout.format == format::any) {
            output_layout.format = impl_param.get_input_layout().format;
        }
        return { output_layout };
    }

    static layout calc_output_layout(custom_gpu_primitive_node const& node, kernel_impl_params const& impl_param) {
        assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
               "Output data type forcing is not supported for "
               "custom_gpu_primitive_node!");
        layout output_layout = impl_param.typed_desc<custom_gpu_primitive>()->output_layout;

        // if the output layout format was set to any, it means the layer output format will be the same as the first
        // input
        if (output_layout.format == format::any) {
            output_layout.format = impl_param.get_input_layout().format;
        }
        return output_layout;
    }

    static std::string to_string(custom_gpu_primitive_node const& node);

public:
    typed_primitive_inst(network& network, custom_gpu_primitive_node const& node);
};

using custom_gpu_primitive_inst = typed_primitive_inst<custom_gpu_primitive>;

}  // namespace cldnn
