// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prior_box_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"

#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "prior_box_clustered_shape_inference.hpp"
#include "prior_box_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(prior_box)

namespace {
template <typename dtype>
void calculate_prior_box_output(memory::ptr output_mem, stream& stream, layout const& input_layout, prior_box& argument) {
    // Calculate output.
    // All the inputs for this layer are known at this point,
    // so the output buffer is written here and not in execute().

    const int layer_width = input_layout.spatial(0);
    const int layer_height = input_layout.spatial(1);
    const int img_width = argument.img_size.spatial[0];
    const int img_height = argument.img_size.spatial[1];
    float step_w = argument.step_width;
    float step_h = argument.step_height;
    if (!argument.is_clustered() && (step_w == 0 || step_h == 0)) {
        step_w = static_cast<float>(img_width) / layer_width;
        step_h = static_cast<float>(img_height) / layer_height;
    }
    const float offset = argument.offset;
    int num_priors = argument.is_clustered() ?
        static_cast<int>(argument.widths.size()) :
        output_mem->get_layout().spatial(1) / 4 / layer_width / layer_height;
    int var_size = static_cast<int>(argument.variance.size());

    mem_lock<dtype> lock{output_mem, stream};
    auto out_ptr = lock.begin();
    int dim = layer_height * layer_width * num_priors * 4;

    int idx = 0;
    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            float center_x, center_y;
            if (argument.step_width == 0.f || argument.step_height == 0.f) {
                center_x = (w + 0.5f) * step_w;
                center_y = (h + 0.5f) * step_h;
            } else {
                center_x = (w + offset) * step_w;
                center_y = (h + offset) * step_h;
            }
            float box_width, box_height;

            if (argument.is_clustered()) {
                for (int s = 0; s < num_priors; ++s) {
                    box_width = argument.widths[s];
                    box_height = argument.heights[s];
                    idx = h * layer_width * num_priors * 4 + w * num_priors * 4 + s * 4;
                    // xmin
                    out_ptr[idx++] = (dtype)((center_x - box_width / 2.f) / img_width);
                    // ymin
                    out_ptr[idx++] = (dtype)((center_y - box_height / 2.f) / img_height);
                    // xmax
                    out_ptr[idx++] = (dtype)((center_x + box_width / 2.f) / img_width);
                    // ymax
                    out_ptr[idx++] = (dtype)((center_y + box_height / 2.f) / img_height);
                }
                continue;
            }

            for (size_t fs = 0; fs < argument.fixed_size.size(); ++fs) {
                auto fixed_size = static_cast<size_t>(argument.fixed_size[fs]);
                auto density = static_cast<size_t>(argument.density[fs]);
                auto shift = fixed_size / density;

                if (argument.fixed_ratio.size() > 0) {
                    for (auto fr : argument.fixed_ratio) {
                        box_width = fixed_size * sqrt(fr);
                        box_height = fixed_size / sqrt(fr);

                        for (size_t r = 0; r < density; ++r) {
                            for (size_t c = 0; c < density; ++c) {
                                float tmp_center_x = center_x - fixed_size / 2.f + shift / 2.f + c * shift;
                                float tmp_center_y = center_y - fixed_size / 2.f + shift / 2.f + r * shift;
                                // xmin
                                out_ptr[idx++] = (dtype)((tmp_center_x - box_width / 2.f) / img_width);
                                // ymin
                                out_ptr[idx++] = (dtype)((tmp_center_y - box_height / 2.f) / img_height);
                                // xmax
                                out_ptr[idx++] = (dtype)((tmp_center_x + box_width / 2.f) / img_width);
                                // ymax
                                out_ptr[idx++] = (dtype)((tmp_center_y + box_height / 2.f) / img_height);
                            }
                        }
                    }
                } else {
                    box_width = box_height = static_cast<float>(fixed_size);

                    for (size_t r = 0; r < density; ++r) {
                        for (size_t c = 0; c < density; ++c) {
                            float tmp_center_x = center_x - fixed_size / 2.f + shift / 2.f + c * shift;
                            float tmp_center_y = center_y - fixed_size / 2.f + shift / 2.f + r * shift;
                            // xmin
                            out_ptr[idx++] = (dtype)((tmp_center_x - box_width / 2.f) / img_width);
                            // ymin
                            out_ptr[idx++] = (dtype)((tmp_center_y - box_height / 2.f) / img_height);
                            // xmax
                            out_ptr[idx++] = (dtype)((tmp_center_x + box_width / 2.f) / img_width);
                            // ymax
                            out_ptr[idx++] = (dtype)((tmp_center_y + box_height / 2.f) / img_height);
                        }
                    }

                    for (auto ar : argument.aspect_ratios) {
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }

                        box_width = fixed_size * sqrt(ar);
                        box_height = fixed_size / sqrt(ar);

                        for (size_t r = 0; r < density; ++r) {
                            for (size_t c = 0; c < density; ++c) {
                                float tmp_center_x = center_x - fixed_size / 2.f + shift / 2.f + c * shift;
                                float tmp_center_y = center_y - fixed_size / 2.f + shift / 2.f + r * shift;
                                // xmin
                                out_ptr[idx++] = (dtype)((tmp_center_x - box_width / 2.f) / img_width);
                                // ymin
                                out_ptr[idx++] = (dtype)((tmp_center_y - box_height / 2.f) / img_height);
                                // xmax
                                out_ptr[idx++] = (dtype)((tmp_center_x + box_width / 2.f) / img_width);
                                // ymax
                                out_ptr[idx++] = (dtype)((tmp_center_y + box_height / 2.f) / img_height);
                            }
                        }
                    }
                }
            }

            for (size_t s = 0; s < argument.min_sizes.size(); ++s) {
                float min_size = argument.min_sizes[s];
                // first prior: aspect_ratio = 1, size = min_size
                box_width = box_height = min_size;
                // xmin
                out_ptr[idx++] = (dtype)((center_x - box_width / 2.f) / img_width);
                // ymin
                out_ptr[idx++] = (dtype)((center_y - box_height / 2.f) / img_height);
                // xmax
                out_ptr[idx++] = (dtype)((center_x + box_width / 2.f) / img_width);
                // ymax
                out_ptr[idx++] = (dtype)((center_y + box_height / 2.f) / img_height);

                if (argument.max_sizes.size() > 0) {
                    float max_size_ = argument.max_sizes[s];
                    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    box_width = box_height = sqrt(min_size * max_size_);
                    // xmin
                    out_ptr[idx++] = (dtype)((center_x - box_width / 2.f) / img_width);
                    // ymin
                    out_ptr[idx++] = (dtype)((center_y - box_height / 2.f) / img_height);
                    // xmax
                    out_ptr[idx++] = (dtype)((center_x + box_width / 2.f) / img_width);
                    // ymax
                    out_ptr[idx++] = (dtype)((center_y + box_height / 2.f) / img_height);
                }

                if (argument.scale_all_sizes || (!argument.scale_all_sizes && (s == argument.min_sizes.size() - 1))) {
                    min_size = argument.scale_all_sizes ? argument.min_sizes[s] : argument.min_sizes[0];
                    // rest of priors
                    for (size_t r = 0; r < argument.aspect_ratios.size(); ++r) {
                        float ar = argument.aspect_ratios[r];
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }
                        box_width = min_size * sqrt(ar);
                        box_height = min_size / sqrt(ar);
                        // xmin
                        out_ptr[idx++] = (dtype)((center_x - box_width / 2.f) / img_width);
                        // ymin
                        out_ptr[idx++] = (dtype)((center_y - box_height / 2.f) / img_height);
                        // xmax
                        out_ptr[idx++] = (dtype)((center_x + box_width / 2.f) / img_width);
                        // ymax
                        out_ptr[idx++] = (dtype)((center_y + box_height / 2.f) / img_height);
                    }
                }
            }
        }
    }

    // clip the prior's coordinate such that it is within [0, 1]
    if (argument.clip) {
        for (int d = 0; d < dim; ++d) {
            out_ptr[d] = (dtype)std::min(std::max(static_cast<float>(out_ptr[d]), 0.f), 1.f);
        }
    }

    // set the variance.
    int count = output_mem->get_layout().spatial(0) * output_mem->get_layout().spatial(1);
    int var_loop_count = argument.is_clustered() ? var_size : 4;
    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            for (int i = 0; i < num_priors; ++i) {
                for (int j = 0; j < var_loop_count; ++j) {
                out_ptr[count] = (dtype)((var_size == 1) ? argument.variance[0] : argument.variance[j]);
                ++count;
                }
            }
        }
    }
}

std::string vector_to_string(const std::vector<float>& vec) {
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
                         const std::vector<float>& densities,
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

    for (auto density : densities) {
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

void prior_box_node::calc_result() {
    if (result != nullptr)
        return;

    auto& argument = *typed_desc();

    // Check arguments
    bool fixed_size_path = !argument.fixed_size.empty() && !argument.density.empty();

    if (!argument.is_clustered() && !fixed_size_path) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(),
                                       "Argument min size",
                                       argument.min_sizes.size(),
                                       "not proper size",
                                       0,
                                       "Must provide at least one min size or fixed_size and density.");
    }

    if (argument.is_clustered()) {
        CLDNN_ERROR_NOT_EQUAL(id(),
            "widths size",
            argument.widths.size(),
            "heights size",
            argument.heights.size(),
            "Clustered prior box requires to have width and height sizes to be equal.");
    }

    for (size_t i = 0; i < argument.min_sizes.size(); i++) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(),
                                       "Min size value at index: " + std::to_string(i),
                                       argument.min_sizes[i],
                                       "less or equal than 0",
                                       0,
                                       "Min size must be positive.");
    }
    if (argument.max_sizes.size() > 0) {
        CLDNN_ERROR_NOT_EQUAL(id(),
                              "Argument min sizes",
                              argument.min_sizes.size(),
                              "argument max sizes",
                              argument.max_sizes.size(),
                              "Number of min sizes must be equal to number of max sizes.");
    }
    for (size_t i = 0; i < argument.max_sizes.size(); i++) {
        CLDNN_ERROR_GREATER_OR_EQUAL_THAN(id(),
                                          "Argument min size value",
                                          argument.min_sizes[i],
                                          "argument max sizes value",
                                          argument.max_sizes[i],
                                          "Max size must be greater than Min size.");
    }
    if (argument.variance.size() > 1) {
        CLDNN_ERROR_NOT_EQUAL(id(),
                              "Argument variance size",
                              argument.variance.size(),
                              "not proper size",
                              4,
                              "Must provide 4 variances.");
        for (size_t i = 0; i < argument.variance.size(); i++) {
            CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(),
                                           "Varaiance value at index: " + std::to_string(i),
                                           argument.variance[i],
                                           "value",
                                           0,
                                           "Variance must be positive.");
        }
    } else if (argument.variance.size() == 1) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(),
                                       "Varaiance value at index 0",
                                       argument.variance[0],
                                       "value",
                                       0,
                                       "Variance must be positive.");
    }

    CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(),
                                   "Image dimension spatial X",
                                   argument.img_size.spatial[0],
                                   "value",
                                   0,
                                   "Image spatial X must be positive.");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(),
                                   "Image dimension spatial Y",
                                   argument.img_size.spatial[1],
                                   "value",
                                   0,
                                   "Image spatial Y must be positive.");

    CLDNN_ERROR_LESS_THAN(id(), "Step height", argument.step_height, "value", 0, "Step height must be positive.");
    CLDNN_ERROR_LESS_THAN(id(), "Step width", argument.step_width, "value", 0, "Step width must be positive.");

    if (!argument.fixed_size.empty()) {
        CLDNN_ERROR_NOT_EQUAL(id(),
                              "Fixed sizes count",
                              argument.fixed_size.size(),
                              "densities count", argument.density.size(),
                              "Number of fixed sizes and densities must be equal.");

        for (size_t fs = 0; fs < argument.fixed_size.size(); ++fs) {
            CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(),
                                           "Fixed size at index" + std::to_string(fs),
                                           argument.fixed_size[fs],
                                           "value",
                                           0,
                                           "Fixed size must be positive.");
            CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(),
                                           "Density at index" + std::to_string(fs),
                                           argument.density[fs],
                                           "value",
                                           0,
                                           "Density must be positive.");
        }
    }

    CLDNN_ERROR_BOOL(id(), "Prior box padding", is_padded(), "Prior-box layer doesn't support output padding.");

    // allocate storage
    result = get_program().get_engine().allocate_memory(get_output_layout());

    // perform calculations
    if (get_output_layout().data_type == data_types::f16)
        calculate_prior_box_output<ov::element_type_traits<data_types::f16>::value_type>(result,
                                                                             get_program().get_stream(),
                                                                             get_input_layout(),
                                                                             *typed_desc());
    else
        calculate_prior_box_output<ov::element_type_traits<data_types::f32>::value_type>(result,
                                                                             get_program().get_stream(),
                                                                             get_input_layout(),
                                                                             *typed_desc());
}

layout prior_box_inst::calc_output_layout(prior_box_node const& node, kernel_impl_params const& impl_param) {
    const auto primitive = impl_param.typed_desc<prior_box>();
    auto number = number_of_priors(primitive->aspect_ratios,
                                   primitive->min_sizes,
                                   primitive->max_sizes,
                                   primitive->fixed_size,
                                   primitive->fixed_ratio,
                                   primitive->density,
                                   primitive->scale_all_sizes,
                                   primitive->flip);
    if (primitive->is_clustered()) {
        number = primitive->widths.size();
    }
    const auto output_type = primitive->output_data_types[0].value_or(data_types::f32);
    const auto output_shape = get_output_shape(primitive->output_size.spatial[1], primitive->output_size.spatial[0], number);

    return {output_type, impl_param.get_input_layout().format, output_shape};
}

template<typename ShapeType>
std::vector<layout> prior_box_inst::calc_output_layouts(prior_box_node const& /*node*/, kernel_impl_params const& impl_param) {
    const auto primitive = impl_param.typed_desc<prior_box>();

    std::vector<ShapeType> input_shapes = {
        impl_param.get_input_layout(0).get<ShapeType>(),
        impl_param.get_input_layout(1).get<ShapeType>()
    };
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::unordered_map<size_t, ov::Tensor> const_data;

    auto& memory_deps = impl_param.memory_deps;

    if (memory_deps.count(0) && memory_deps.count(1)) {
        auto output_size_mem = memory_deps.at(0);
        auto img_size_mem = memory_deps.at(1);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> output_size_lock(output_size_mem, impl_param.get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> img_size_lock(img_size_mem, impl_param.get_stream());

        const_data.emplace(0, make_tensor(output_size_mem->get_layout(), output_size_lock.data()));

        auto p_param = const_cast<kernel_impl_params*>(&impl_param);
        if (output_size_mem->get_layout().data_type == cldnn::data_types::i64) {
            auto output_height = reinterpret_cast<int64_t*>(output_size_lock.data())[0];
            auto output_width = reinterpret_cast<int64_t*>(output_size_lock.data())[1];
            auto img_height = reinterpret_cast<int64_t*>(img_size_lock.data())[0];
            auto img_width = reinterpret_cast<int64_t*>(img_size_lock.data())[1];

            if (p_param->output_size.empty()) {
                p_param->output_size.push_back(static_cast<size_t>(output_width));
                p_param->output_size.push_back(static_cast<size_t>(output_height));
            } else {
                p_param->output_size[0] = static_cast<size_t>(output_width);
                p_param->output_size[1] = static_cast<size_t>(output_height);
            }

            if (p_param->img_size.empty()) {
                p_param->img_size.push_back(static_cast<size_t>(img_width));
                p_param->img_size.push_back(static_cast<size_t>(img_height));
            } else {
                p_param->img_size[0] = static_cast<size_t>(img_width);
                p_param->img_size[1] = static_cast<size_t>(img_height);
            }
        } else { //int32_t
            auto output_height = reinterpret_cast<int32_t*>(output_size_lock.data())[0];
            auto output_width = reinterpret_cast<int32_t*>(output_size_lock.data())[1];
            auto img_height = reinterpret_cast<int32_t*>(img_size_lock.data())[0];
            auto img_width = reinterpret_cast<int32_t*>(img_size_lock.data())[1];

            if (p_param->output_size.empty()) {
                p_param->output_size.push_back(static_cast<size_t>(output_width));
                p_param->output_size.push_back(static_cast<size_t>(output_height));
            } else {
                p_param->output_size[0] = static_cast<size_t>(output_width);
                p_param->output_size[1] = static_cast<size_t>(output_height);
            }

            if (p_param->img_size.empty()) {
                p_param->img_size.push_back(static_cast<size_t>(img_width));
                p_param->img_size.push_back(static_cast<size_t>(img_height));
            } else {
                p_param->img_size[0] = static_cast<size_t>(img_width);
                p_param->img_size[1] = static_cast<size_t>(img_height);
            }
        }
    }

    const auto tensor_accessor = ov::make_tensor_accessor(const_data);
    if (primitive->is_clustered()) {
        ov::op::v0::PriorBoxClustered op;
        op.set_attrs(primitive->get_attrs_clustered());
        output_shapes = ov::op::v0::shape_infer(&op, input_shapes, tensor_accessor);
    } else {
        if (primitive->is_v8_support()) {
            ov::op::v8::PriorBox op;
            op.set_attrs(primitive->get_attrs_v8());
            output_shapes = ov::op::v8::shape_infer(&op, input_shapes, tensor_accessor);
        } else {
            ov::op::v0::PriorBox op;
            op.set_attrs(primitive->get_attrs_v0());
            output_shapes = ov::op::v0::shape_infer(&op, input_shapes, tensor_accessor);
        }
    }
    const auto output_type = primitive->output_data_types[0].value_or(data_types::f32);

    return {layout{output_shapes[0], output_type, impl_param.get_input_layout().format}};
}

template std::vector<layout> prior_box_inst::calc_output_layouts<ov::PartialShape>(prior_box_node const& /*node*/, kernel_impl_params const& impl_param);

std::string prior_box_inst::to_string(prior_box_node const& node) {
    auto desc = node.get_primitive();
    auto flip = desc->flip ? "true" : "false";
    auto clip = desc->clip ? "true" : "false";
    auto scale_all_sizes = desc->scale_all_sizes ? "true" : "false";
    auto node_info = node.desc_to_json();

    std::string str_min_sizes = vector_to_string(desc->min_sizes);
    std::string str_max_sizes = vector_to_string(desc->max_sizes);
    std::string str_variance = vector_to_string(desc->variance);
    std::string str_aspect_ratio = vector_to_string(desc->aspect_ratios);
    std::string str_fixed_size = vector_to_string(desc->fixed_size);
    std::string str_fixed_ratio = vector_to_string(desc->fixed_ratio);
    std::string str_density = vector_to_string(desc->density);

    std::stringstream primitive_description;

    json_composite prior_info;
    prior_info.add("input id", node.input().id());
    prior_info.add("iamge size", desc->img_size);
    prior_info.add("variance", std::move(str_variance));

    json_composite box_sizes_info;
    box_sizes_info.add("min sizes", std::move(str_min_sizes));
    box_sizes_info.add("max sizes", std::move(str_max_sizes));
    prior_info.add("box sizes", box_sizes_info);

    prior_info.add("aspect_ratio", str_aspect_ratio);
    prior_info.add("flip", flip);
    prior_info.add("clip", clip);
    prior_info.add("scale all sizes", scale_all_sizes);
    prior_info.add("fixed size", std::move(str_fixed_size));
    prior_info.add("fixed ratio", str_fixed_ratio);
    prior_info.add("density", str_density);

    json_composite step_info;
    step_info.add("step width", desc->step_width);
    step_info.add("step height", desc->step_height);
    step_info.add("offset", desc->offset);
    prior_info.add("step", step_info);
    prior_info.add("min max aspect ratios order", desc->min_max_aspect_ratios_order);

    if (node.is_clustered()) {
        json_composite clustered_info;
        step_info.add("widths", desc->widths);
        step_info.add("heights", desc->heights);
        prior_info.add("clustered info", clustered_info);
    }

    node_info->add("prior box info", prior_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

prior_box_inst::typed_primitive_inst(network& network, prior_box_node const& node) : parent(network, node) {
}

}  // namespace cldnn
