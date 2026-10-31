// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "primitive_inst.h"
#include "openvino/op/parameter.hpp"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<custom_gpu_primitive> : public typed_program_node_base<custom_gpu_primitive> {
    using parent = typed_program_node_base<custom_gpu_primitive>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog) : parent(prim, prog) {}
    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
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
        std::vector<layout> output_layouts = impl_param.typed_desc<custom_gpu_primitive>()->output_layouts;

        typed_primitive_inst<custom_gpu_primitive>::update_output_shape(impl_param, output_layouts);

        // if the output layout format was set to any, it means the layer output format will be the same as the first input
        for (auto& output_layout : output_layouts) {
            if (output_layout.format == format::any) {
                output_layout.format = impl_param.get_input_layout().format;
            }
        }
        return output_layouts;
    }

    static layout calc_output_layout(custom_gpu_primitive_node const& node, kernel_impl_params const& impl_param) {
        assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
               "Output data type forcing is not supported for "
               "custom_gpu_primitive_node!");
        auto output_layouts = impl_param.typed_desc<custom_gpu_primitive>()->output_layouts;

        typed_primitive_inst<custom_gpu_primitive>::update_output_shape(impl_param, output_layouts);

        // if the output layout format was set to any, it means the layer output format will be the same as the first
        // input
        if (output_layouts[0].format == format::any) {
            output_layouts[0].format = impl_param.get_input_layout().format;
        }
        return output_layouts[0];
    }

    static std::string to_string(custom_gpu_primitive_node const& node);

public:
    typed_primitive_inst(network& network, custom_gpu_primitive_node const& node);

private:
    static void update_output_shape(const kernel_impl_params& impl_param, std::vector<layout>& output_layouts) {
        bool is_dynamic_input = false;
        const auto inp_sz = impl_param.get_input_layout_size();
        for (size_t i = 0; i < inp_sz; i++) {
            if (impl_param.get_input_layout(i).is_dynamic()) {
                is_dynamic_input = true;
                break;
            }
        }

        bool is_dynamic_output = false;
        for (size_t i = 0; i < output_layouts.size(); i++) {
            if (output_layouts[i].is_dynamic()) {
                is_dynamic_output = true;
                break;
            }
        }

        // Execute the op's shape inference only for dynamic node when input shapes have already been calculated; otherwise, keep the original output layout
        // unchanged (it will be either static for static model or have dynamic shape in case of dynamic flow)
        if (!is_dynamic_input && is_dynamic_output) {
            ov::OutputVector new_inputs;
            for (size_t i = 0; i < inp_sz; i++) {
                auto input = std::make_shared<ov::op::v0::Parameter>(impl_param.get_input_layout(i).data_type, impl_param.get_input_layout(i).get_shape());
                new_inputs.emplace_back(input);
            }

            auto op = impl_param.typed_desc<custom_gpu_primitive>()->op;
            if (!op) {  // model loaded from the cache: the original op is not serialized, use the recorded shape rule
                const auto prim = impl_param.typed_desc<custom_gpu_primitive>();
                const auto& src = prim->output_shape_src;
                const auto& sdims = prim->output_static_dims;
                OPENVINO_ASSERT(src.size() == output_layouts.size(),
                                "[GPU] Custom layer ", prim->id, " loaded from the cache has no output shape rule");
                size_t off = 0;
                for (size_t i = 0; i < output_layouts.size(); i++) {
                    OPENVINO_ASSERT(src[i] >= 0 && static_cast<size_t>(src[i]) < inp_sz,
                                    "[GPU] Custom layer ", prim->id, " loaded from the cache: output ", i,
                                    " has no input to take its dynamic dims from");
                    const auto in_shape = impl_param.get_input_layout(src[i]).get_shape();
                    const size_t rank = output_layouts[i].get_partial_shape().size();
                    OPENVINO_ASSERT(off + rank <= sdims.size(),
                                    "[GPU] Custom layer ", prim->id, " loaded from the cache: no static dims for output ", i);
                    ov::Shape s(rank);
                    for (size_t d = 0; d < rank; d++) {
                        const int64_t v = sdims[off + d];
                        OPENVINO_ASSERT(v >= 0 || d < in_shape.size(),
                                        "[GPU] Custom layer ", prim->id, " loaded from the cache: bad dim ", d, " of output ", i);
                        s[d] = v >= 0 ? static_cast<size_t>(v) : in_shape[d];
                    }
                    off += rank;
                    output_layouts[i].set_partial_shape(s);
                }
                return;
            }
            auto new_op = op->clone_with_new_inputs(new_inputs);
            new_op->validate_and_infer_types();
            OPENVINO_ASSERT(output_layouts.size() == new_op->get_output_size(), "Output size should keep align.");
            for (size_t i = 0; i < output_layouts.size(); i++) {
                auto new_outp_shape = new_op->get_output_shape(i);
                output_layouts[i].set_partial_shape(new_outp_shape);
            }
        }
    }
};

using custom_gpu_primitive_inst = typed_primitive_inst<custom_gpu_primitive>;

}  // namespace cldnn
