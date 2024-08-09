// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/runtime/memory.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "multinomial_inst.h"
#include "multinomial_shape_inference.hpp"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(multinomial)

template<typename ShapeType>
std::vector<layout> multinomial_inst::calc_output_layouts(multinomial_node const& /*node*/, const kernel_impl_params& impl_param) {
    const auto& input_layout = impl_param.get_input_layout();
    auto primitive = impl_param.typed_desc<multinomial>();

    layout out_layout{ov::PartialShape{input_layout.get_partial_shape()[0], primitive->num_samples}, primitive->output_data_type, input_layout.format};

    return { out_layout };
}

template std::vector<layout> multinomial_inst::calc_output_layouts<ov::PartialShape>(multinomial_node const& node, const kernel_impl_params& impl_param);

layout multinomial_inst::calc_output_layout(multinomial_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<multinomial>();
    auto input_layout = impl_param.get_input_layout(0);
    if (input_layout.get_shape().size() == 1) {
        return {primitive->output_data_type, input_layout.format,
            tensor{std::vector<tensor::value_type>{
                static_cast<tensor::value_type>(primitive->num_samples)
            }}};
    } else {
        return {primitive->output_data_type, input_layout.format,
            tensor{std::vector<tensor::value_type>{
                input_layout.batch(),
                static_cast<tensor::value_type>(primitive->num_samples)
            }}};
    }
}

multinomial_inst::typed_primitive_inst(network& network, multinomial_node const& node)
    : parent{network, node} {}

std::string multinomial_inst::to_string(multinomial_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& cdf = node.input(0);
    auto& random_probabilities = node.input(1);

    std::stringstream primitive_description;

    json_composite multinomial_info;
    multinomial_info.add("cdf", cdf.id());
    multinomial_info.add("random probabilities", random_probabilities.id());
    multinomial_info.add("output data_type", desc->output_data_type);
    multinomial_info.add("with replacement", desc->with_replacement);
    multinomial_info.add("log probs", desc->log_probs);
    multinomial_info.add("global seed", desc->global_seed);
    multinomial_info.add("op seed", desc->op_seed);
    multinomial_info.add("num samples", desc->num_samples);

    node_info->add("multinomial info", multinomial_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
