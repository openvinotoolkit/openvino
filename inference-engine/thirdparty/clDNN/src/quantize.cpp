/*
// Copyright (c) 2019 Intel Corporation
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

#include "quantize_inst.h"
#include "binary_convolution_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"
#include "error_handler.h"
#include "json_object.h"
#include "data_inst.h"
#include <string>

namespace cldnn {
primitive_type_id quantize::type_id() {
    static primitive_type_base<quantize> instance;
    return &instance;
}

layout quantize_inst::calc_output_layout(quantize_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto output_format = input_layout.format;
    auto out_dt = input_layout.data_type;
    if (node.get_primitive()->output_data_type)
        out_dt = *node.get_primitive()->output_data_type;

    if (out_dt == data_types::bin) {
        output_format = format::b_fs_yx_32fp;
    }

    return layout{out_dt, output_format, input_layout.size};
}

std::string quantize_inst::to_string(quantize_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input(0);
    auto& input_low = node.input(1);
    auto& input_high = node.input(2);
    auto& output_low = node.input(3);
    auto& output_high = node.input(4);
    auto scale_shift_opt = node.get_scale_shift_opt() ? "true" : "false";

    std::stringstream primitive_description;

    json_composite quantize_info;
    quantize_info.add("input id", input.id());
    quantize_info.add("input low id", input_low.id());
    quantize_info.add("input high id", input_high.id());
    quantize_info.add("output low id", output_low.id());
    quantize_info.add("output high id", output_high.id());
    quantize_info.add("scale_shift_opt", scale_shift_opt);
    quantize_info.add("levels", desc->levels);

    node_info->add("quantize info", quantize_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

quantize_inst::typed_primitive_inst(network_impl& network, quantize_node const& node) : parent(network, node) {}

}  // namespace cldnn
