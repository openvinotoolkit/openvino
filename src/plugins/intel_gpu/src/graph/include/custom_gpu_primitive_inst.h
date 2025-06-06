// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "primitive_inst.h"
#include "openvino/op/parameter.hpp"

#include <string>

namespace cldnn {

using custom_gpu_primitive_node = typed_program_node<custom_gpu_primitive>;

template <>
class typed_primitive_inst<custom_gpu_primitive> : public typed_primitive_inst_base<custom_gpu_primitive> {
    using parent = typed_primitive_inst_base<custom_gpu_primitive>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(custom_gpu_primitive_node const& node, const kernel_impl_params& impl_param) {
        assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
               "Output data type forcing is not supported for "
               "custom_gpu_primitive_node!");
        layout output_layout = impl_param.typed_desc<custom_gpu_primitive>()->output_layout;

        bool is_dynamic = false;
        const auto inp_sz = impl_param.get_input_layout_size();
        for (size_t i = 0; i < inp_sz; i++) {
            if (impl_param.get_input_layout(i).is_dynamic()) {
                is_dynamic = true;
                break;
            }
        }

        if (!is_dynamic && output_layout.is_dynamic()) {
            ov::OutputVector new_inputs;
            for (size_t i = 0; i < inp_sz; i++) {
                auto dt = impl_param.get_input_layout(i).data_type;
                // std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0])
                auto input = std::make_shared<ov::op::v0::Parameter>(dt, impl_param.get_input_layout(i).get_shape());
                new_inputs.emplace_back(input);
            }

            auto op = impl_param.typed_desc<custom_gpu_primitive>()->op;
            auto new_op = op->clone_with_new_inputs(new_inputs);
            new_op->validate_and_infer_types();
            auto new_outp_shape = new_op->get_output_shape(0);
            output_layout.set_partial_shape(new_outp_shape);
        }

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

    static kernel_impl_params get_fake_aligned_params(kernel_impl_params const& orig_impl_param);

public:
    typed_primitive_inst(network& network, custom_gpu_primitive_node const& node);
};

using custom_gpu_primitive_inst = typed_primitive_inst<custom_gpu_primitive>;

}  // namespace cldnn
