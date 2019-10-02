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

#include "reduce_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include "data_inst.h"
#include <vector>
#include <string>

namespace cldnn {
primitive_type_id reduce::type_id() {
    static primitive_type_base<reduce> instance;
    return &instance;
}

layout reduce_inst::calc_output_layout(reduce_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;
    auto output_type = input_layout.data_type;
    auto mode = desc->mode;
    auto reduce_axes = desc->axes;

    auto in_dims = input_layout.size.sizes();
    for (size_t a = 0; a < reduce_axes.size(); a++) {
        in_dims[reduce_axes[a]] = 1;
    }

    if (!desc->keep_dims) {
        for (size_t a = 0; a < reduce_axes.size(); a++) {
            in_dims.erase(in_dims.begin() + reduce_axes[a]);
            in_dims.push_back(1);
        }
    }

    std::vector<reduce_mode> reduce_bool_modes = {reduce_mode::logical_and, reduce_mode::logical_or};
    if (std::find(reduce_bool_modes.begin(), reduce_bool_modes.end(), mode) != reduce_bool_modes.end()) output_type = data_types::i8;

    if (input_layout.format == format::bfwzyx)
        return layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3], in_dims[4], in_dims[5]))};
    else if (input_layout.format == format::bfzyx)
        return layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3], in_dims[4]))};
    else
        return layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3]))};
}

std::string reduce_inst::to_string(reduce_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite reduce_info;
    reduce_info.add("input id", node.input(0).id());
    reduce_info.add("axes", desc->axes);
    reduce_info.add("keep_dims", desc->keep_dims);
    reduce_info.add("mode", static_cast<uint16_t>(desc->mode));

    node_info->add("reduce info", reduce_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reduce_inst::typed_primitive_inst(network_impl& network, reduce_node const& node) : parent(network, node) {}

}  // namespace cldnn
