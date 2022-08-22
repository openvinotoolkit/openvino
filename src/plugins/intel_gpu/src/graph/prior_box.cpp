// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <json_object.h>

#include "primitive_type_base.h"
#include "prior_box_inst.h"

namespace {
std::string vector_to_string(std::vector<float> vec) {
    std::stringstream result;
    for (size_t i = 0; i < vec.size(); i++)
        result << vec.at(i) << ", ";
    return result.str();
}

std::vector<float> normalized_aspect_ratio(const std::vector<float>& aspect_ratio, bool flip) {
    std::set<float> unique_ratios;
    for (auto ratio : aspect_ratio) {
        unique_ratios.insert(std::round(ratio * 1e6) / 1e6);
        if (flip)
            unique_ratios.insert(std::round(1 / ratio * 1e6) / 1e6);
    }
    unique_ratios.insert(1);
    return std::vector<float>(unique_ratios.begin(), unique_ratios.end());
}

int64_t number_of_priors(const std::vector<float>& aspect_ratio,
                         const std::vector<float>& min_size,
                         const std::vector<float>& max_size,
                         const std::vector<float>& fixed_size,
                         const std::vector<float>& fixed_ratio,
                         const std::vector<float>& density,
                         bool scale_all_sizes,
                         bool flip) {
    // Starting with 0 number of prior and then various conditions on attributes will contribute
    // real number of prior boxes as PriorBox is a fat thing with several modes of
    // operation that will be checked in order in the next statements.
    int64_t num_priors = 0;

    // Total number of boxes around each point; depends on whether flipped boxes are included
    // plus one box 1x1.
    int64_t total_aspect_ratios = normalized_aspect_ratio(aspect_ratio, flip).size();

    if (scale_all_sizes) {
        num_priors = total_aspect_ratios * min_size.size() + max_size.size();
    } else {
        num_priors = total_aspect_ratios + min_size.size() - 1;
    }

    if (!fixed_size.empty()) {
        num_priors = total_aspect_ratios * fixed_size.size();
    }

    for (auto density : density) {
        auto rounded_density = static_cast<int64_t>(density);
        auto density_2d = (rounded_density * rounded_density - 1);
        if (!fixed_ratio.empty()) {
            num_priors += fixed_ratio.size() * density_2d;
        } else {
            num_priors += total_aspect_ratios * density_2d;
        }
    }
    return num_priors;
}

tensor get_output_shape(int32_t height, int32_t width, int32_t number_of_priors) {
    return tensor{std::vector<int32_t>{2, 4 * height * width * number_of_priors}};
}
}  // namespace

namespace cldnn {
primitive_type_id prior_box::type_id() {
    static primitive_type_base<prior_box> instance;
    return &instance;
}

prior_box_node::typed_program_node(std::shared_ptr<prior_box> prim, program& prog) : parent(prim, prog) {
    constant = true;
}

layout prior_box_inst::calc_output_layout(prior_box_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = node.get_primitive();
    const auto attrs = primitive->attributes;
    int64_t number = number_of_priors(attrs.aspect_ratio,
                                      attrs.min_size,
                                      attrs.max_size,
                                      attrs.fixed_size,
                                      attrs.fixed_ratio,
                                      attrs.density,
                                      attrs.scale_all_sizes,
                                      attrs.flip);
    if (primitive->is_clustered()) {
        number = primitive->attributes.widths.size();
    }
    auto output_shape = get_output_shape(primitive->height, primitive->width, number);

    return {*(primitive->output_data_type), node.input().get_output_layout().format, output_shape};
}

std::string prior_box_inst::to_string(prior_box_node const& node) {
    auto desc = node.get_primitive();
    auto flip = desc->attributes.flip ? "true" : "false";
    auto clip = desc->attributes.clip ? "true" : "false";
    auto scale_all_sizes = desc->attributes.scale_all_sizes ? "true" : "false";
    auto node_info = node.desc_to_json();

    std::string str_min_sizes = vector_to_string(desc->attributes.min_size);
    std::string str_max_sizes = vector_to_string(desc->attributes.max_size);
    std::string str_variance = vector_to_string(desc->variance);
    std::string str_aspect_ratio = vector_to_string(desc->aspect_ratios);
    std::string str_fixed_size = vector_to_string(desc->attributes.fixed_size);
    std::string str_fixed_ratio = vector_to_string(desc->attributes.fixed_ratio);
    std::string str_density = vector_to_string(desc->attributes.density);

    std::stringstream primitive_description;

    json_composite prior_info;
    prior_info.add("input id", node.input().id());
    prior_info.add("variance", str_variance);

    json_composite box_sizes_info;
    box_sizes_info.add("min sizes", str_min_sizes);
    box_sizes_info.add("max sizes", str_max_sizes);
    prior_info.add("box sizes", box_sizes_info);

    prior_info.add("aspect_ratio", str_aspect_ratio);
    prior_info.add("flip", flip);
    prior_info.add("clip", clip);
    prior_info.add("scale all sizes", scale_all_sizes);
    prior_info.add("fixed size", str_fixed_size);
    prior_info.add("fixed ratio", str_fixed_ratio);
    prior_info.add("density", str_density);

    json_composite step_info;
    step_info.add("step width", desc->step_x);
    step_info.add("step height", desc->step_y);
    step_info.add("offset", desc->attributes.offset);
    prior_info.add("step", step_info);

    node_info->add("prior box info", prior_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

prior_box_inst::typed_primitive_inst(network& network, prior_box_node const& node) : parent(network, node) {}

}  // namespace cldnn
