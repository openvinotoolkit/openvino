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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "upsampling_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include <string>

namespace cldnn {
primitive_type_id upsampling_type_id() {
    static primitive_type_base<upsampling> instance;
    return &instance;
}

layout upsampling_inst::calc_output_layout(upsampling_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for upsampling_node!");
    auto desc = node.get_primitive();
    auto input_layout = node.input().get_output_layout();
    auto scale = desc->scale;

    auto result_sizes = tensor(input_layout.size.batch[0],
                               input_layout.size.feature[0],
                               static_cast<size_t>(input_layout.size.spatial[0] * scale),
                               static_cast<size_t>(input_layout.size.spatial[1] * scale));
    auto result = layout({input_layout.data_type, input_layout.format, result_sizes});

    return result;
}

std::string upsampling_inst::to_string(upsampling_node const& node) {
    std::stringstream primitive_description;
    auto desc = node.get_primitive();
    auto& input_1 = node.input();
    auto activation = desc->with_activation ? " true" : "false";
    std::string str_type;
    switch (desc->sample_type) {
        case upsampling_sample_type::nearest:
            str_type = "nearest";
            break;
        case upsampling_sample_type::bilinear:
            str_type = "bilinear";
            break;
        default:
            str_type = "not supported sample type";
            break;
    }

    primitive_description << "id: " << desc->id << ", type: upsampling"
                          << "\n\tinput_1: " << input_1.id() << ", count: " << input_1.get_output_layout().count()
                          << ",  size: " << input_1.get_output_layout().size << "\n\tscale: " << desc->scale
                          << "\n\tnum_filter: " << desc->num_filter << "\n\tsample_type: " << str_type
                          << "\n\twith activation: " << activation << ", slope: " << desc->activation_negative_slope
                          << "\n\toutput padding lower size: " << desc->output_padding.lower_size()
                          << "\n\toutput padding upper size: " << desc->output_padding.upper_size()
                          << "\n\toutput: count: " << node.get_output_layout().count()
                          << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

upsampling_inst::typed_primitive_inst(network_impl& network, upsampling_node const& node) : parent(network, node) {
    if (argument.sample_type == upsampling_sample_type::bilinear)
        CLDNN_ERROR_MESSAGE(
            node.id(),
            "Upsampling primitive instance with bilinear filtering should be replaced by deconvolution!");
}
}  // namespace cldnn
