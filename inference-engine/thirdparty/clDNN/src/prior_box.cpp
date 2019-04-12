/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "prior_box_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

#include <cmath>

namespace cldnn
{
primitive_type_id prior_box_type_id()
{
    static primitive_type_base<prior_box> instance;
    return &instance;
}

namespace {
    template<typename dtype>
    void calculate_prior_box_output(memory_impl& output_mem, layout const& input_layout, prior_box& argument)
    {
        // Calculate output.
        // All the inputs for this layer are known at this point,
        // so the output buffer is written here and not in execute().

        const int layer_width = input_layout.size.spatial[0];
        const int layer_height = input_layout.size.spatial[1];
        const int img_width = argument.img_size.spatial[0];
        const int img_height = argument.img_size.spatial[1];
        float step_w = argument.step_width;
        float step_h = argument.step_height;
        if (step_w == 0 || step_h == 0) {
            step_w = static_cast<float>(img_width) / layer_width;
            step_h = static_cast<float>(img_height) / layer_height;
        }
        const float offset = argument.offset;
        int num_priors = argument.scale_all_sizes ? (int)argument.aspect_ratios.size() * (int)argument.min_sizes.size()
                                                                                       + (int)argument.max_sizes.size()
                                                  : (int)argument.aspect_ratios.size() + (int)argument.min_sizes.size()
                                                                                       + (int)argument.max_sizes.size() - 1;

        mem_lock<dtype> lock{ output_mem };
        auto out_ptr = lock.begin();

        int dim = layer_height * layer_width * num_priors * 4;
        int idx = 0;
        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                float center_x = (w + offset) * step_w;
                float center_y = (h + offset) * step_h;
                float box_width, box_height;
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

                    if (argument.scale_all_sizes || (!argument.scale_all_sizes && (s == argument.min_sizes.size() - 1)))
                    {
                        min_size = argument.scale_all_sizes ? argument.min_sizes[s] : argument.min_sizes[0];
                        // rest of priors
                        for (size_t r = 0; r < argument.aspect_ratios.size(); ++r)
                        {
                            float ar = argument.aspect_ratios[r];
                            if (fabs(ar - 1.) < 1e-6)
                            {
                                continue;
                            }
                            box_width = min_size * sqrt(ar);
                            box_height = min_size / sqrt(ar);
                            // xmin
                            out_ptr[idx++] = (dtype) ((center_x - box_width / 2.f) / img_width);
                            // ymin
                            out_ptr[idx++] = (dtype) ((center_y - box_height / 2.f) / img_height);
                            // xmax
                            out_ptr[idx++] = (dtype) ((center_x + box_width / 2.f) / img_width);
                            // ymax
                            out_ptr[idx++] = (dtype) ((center_y + box_height / 2.f) / img_height);
                        }
                    }
                }
            }
        }

        // clip the prior's coordinate such that it is within [0, 1]
        if (argument.clip) {
            for (int d = 0; d < dim; ++d) {
                out_ptr[d] = (dtype)std::min(std::max((float)out_ptr[d], 0.f), 1.f);
            }
        }

        // set the variance.
        int count = output_mem.get_layout().size.spatial[0] * output_mem.get_layout().size.spatial[1];
        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                for (int i = 0; i < num_priors; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        out_ptr[count] = (dtype)((argument.variance.size() == 1) ? argument.variance[0] : argument.variance[j]);
                        ++count;
                    }
                }
            }
        }
    }
}

prior_box_node::typed_program_node(std::shared_ptr<prior_box> prim, program_impl& prog)
    : parent(prim, prog)
{
    constant = true;
}

void prior_box_node::calc_result()
{
    if (result != nullptr)
        return;

    auto& argument = *typed_desc();

    //Check arguments
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(), "Argument min size", argument.min_sizes.size(), "not proper size", 0, "Must provide at least one min size.");

    for (size_t i = 0; i < argument.min_sizes.size(); i++) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(), "Min size value at index: " + std::to_string(i), argument.min_sizes[i], "less or equal than 0", 0, "Min size must be positive.");
    }
    if (argument.max_sizes.size() > 0) {
        CLDNN_ERROR_NOT_EQUAL(id(), "Argument min sizes", argument.min_sizes.size(), "argument max sizes", argument.max_sizes.size(), "Number of min sizes must be equal to number of max sizes.");
    }
    for (size_t i = 0; i < argument.max_sizes.size(); i++) {
        CLDNN_ERROR_GREATER_OR_EQUAL_THAN(id(), "Argument min size value", argument.min_sizes[i], "argument max sizes value", argument.max_sizes[i], "Max size must be greater than Min size.");
    }
    if (argument.variance.size() > 1) {
        CLDNN_ERROR_NOT_EQUAL(id(), "Argument variance size", argument.variance.size(), "not proper size", 4, "Must provide 4 variances.");
        for (size_t i = 0; i < argument.variance.size(); i++) {
            CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(), "Varaiance value at index: " + std::to_string(i), argument.variance[i], "value", 0, "Variance must be positive.");
        }
    }
    else if (argument.variance.size() == 1) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(), "Varaiance value at index 0", argument.variance[0], "value", 0, "Variance must be positive.");
    }

    CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(), "Image dimension spatial X", argument.img_size.spatial[0], "value", 0, "Image spatial X must be positive.");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(id(), "Image dimension spatial Y", argument.img_size.spatial[1], "value", 0, "Image spatial Y must be positive.");

    CLDNN_ERROR_LESS_THAN(id(), "Step height", argument.step_height, "value", 0, "Step height must be positive.");
    CLDNN_ERROR_LESS_THAN(id(), "Step width", argument.step_width, "value", 0, "Step width must be positive.");

    CLDNN_ERROR_BOOL(id(), "Prior box padding", is_padded(), "Prior-box layer doesn't support output padding.");

    //allocate storage
    result = get_program().get_engine().allocate_memory(get_output_layout());

    //perform calculations
    if (input().get_output_layout().data_type == data_types::f16)
        calculate_prior_box_output<data_type_to_type<data_types::f16>::type>(*result, input().get_output_layout(), *typed_desc());
    else
        calculate_prior_box_output<data_type_to_type<data_types::f32>::type>(*result, input().get_output_layout(), *typed_desc());
}

layout prior_box_inst::calc_output_layout(prior_box_node const& node)
{
    assert((bool)node.get_primitive()->output_data_type == false
           && "Output data type forcing is not supported for prior_box_node!");
    auto desc = node.get_primitive();
    auto input_layout = node.input().get_output_layout();
    assert(input_layout.size.spatial.size() == 2);

    const int layer_width = input_layout.size.spatial[0];
    const int layer_height = input_layout.size.spatial[1];

    int num_priors = desc->scale_all_sizes ? (int)desc->aspect_ratios.size() * (int)desc->min_sizes.size() + (int)desc->max_sizes.size()
                                           : (int)desc->aspect_ratios.size() + (int)desc->min_sizes.size() + (int)desc->max_sizes.size() - 1;

    // Since all images in a batch has same height and width, we only need to
    // generate one set of priors which can be shared across all images.
    // 2 features. First feature stores the mean of each prior coordinate.
    // Second feature stores the variance of each prior coordinate.

    auto output_data_type = input_layout.data_type == data_types::f16 ? data_types::f16 : data_types::f32;
    return{ output_data_type, cldnn::format::bfyx, cldnn::tensor( 1, 2, 1, layer_width * layer_height * num_priors * 4 ) };
}

std::string vector_to_string(std::vector<float> vec)
{
    std::stringstream result;
    for (size_t i = 0; i < vec.size(); i++)
        result << vec.at(i) << ", ";
    return result.str();
}

std::string prior_box_inst::to_string(prior_box_node const& node)
{
    auto desc            = node.get_primitive();
    auto flip            = desc->flip ? "true" : "false";
    auto clip            = desc->clip ? "true" : "false";
    auto scale_all_sizes = desc->scale_all_sizes ? "true" : "false";
    auto node_info       = node.desc_to_json();

    std::string str_min_sizes    = vector_to_string(desc->min_sizes);
    std::string str_max_sizes    = vector_to_string(desc->max_sizes);
    std::string str_variance     = vector_to_string(desc->variance);
    std::string str_aspect_ratio = vector_to_string(desc->aspect_ratios);
    
    std::stringstream primitive_description;

    json_composite prior_info;
    prior_info.add("input id", node.input().id());
    prior_info.add("iamge size", desc->img_size);
    prior_info.add("variance", str_variance);

    json_composite box_sizes_info;
    box_sizes_info.add("min sizes", str_min_sizes);
    box_sizes_info.add("max sizes", str_max_sizes);
    prior_info.add("box sizes", box_sizes_info);

    prior_info.add("aspect_ratio", str_aspect_ratio);
    prior_info.add("flip", flip);
    prior_info.add("clip", clip);
    prior_info.add("scale all sizes", scale_all_sizes);

    json_composite step_info;
    step_info.add("step width", desc->step_width);
    step_info.add("step height", desc->step_height);
    step_info.add("offset", desc->offset);
    prior_info.add("step", step_info);

    node_info->add("prior box info", prior_info);
    node_info->dump(primitive_description);
    
    return primitive_description.str();
}

prior_box_inst::typed_primitive_inst(network_impl& network, prior_box_node const& node)
    :parent(network, node)
{
    CLDNN_ERROR_MESSAGE(node.id(), "Prior box primitive instance should not be created!");
}

}
