/*
// Copyright (c) 2016-2019 Intel Corporation
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
#include "fully_connected_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id fully_connected::type_id() {
    static primitive_type_base<fully_connected> instance;
    return &instance;
}

namespace {
bool is_batch_after_spatial(const std::string order) {
    bool spatial_found = false;
    for (auto c : order) {
        switch (c) {
            case 'b':
            case 'n':
                return spatial_found;

            case 'x':
            case 'y':
            case 'z':
            case 'w':
            case 's':
                spatial_found = true;
                break;

            default:
                break;
        }
    }
    return false;
}
}  // namespace

layout fully_connected_inst::calc_output_layout(fully_connected_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for "
           "fully_connected_node!");
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights().get_output_layout();

    if (is_batch_after_spatial(input_layout.format.order()) ||
        (input_layout.format ==
             format::bfyx &&  // this condition tests whether our input is batch>1 in bfyx format, if yes there will be
         input_layout.size.batch[0] > 1) ||  // extra reorder between input and this fc from bfyx to yxfb format (so
                                             // "is_batch_after_spatial" should return true)
        input_layout.format == format::bs_x_bsv16 ||
        input_layout.format == format::bs_xs_xsv8_bsv8) {
        auto result = layout(input_layout.data_type,
                             format::yxfb,
                             tensor(input_layout.size.batch[0], weights_layout.size.batch[0], 1, 1));
        return result;
    } else {
        auto result = layout(input_layout.data_type,
                             format::bfyx,
                             tensor(input_layout.size.batch[0], weights_layout.size.batch[0], 1, 1));
        return result;
    }
}

std::string fully_connected_inst::to_string(fully_connected_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto bias_id = desc->bias != "" ? desc->bias : "no bias";
    auto weights_id = desc->weights;

    std::stringstream primitive_description;

    json_composite fc_info;
    fc_info.add("weights id", weights_id);
    fc_info.add("bias id", bias_id);

    node_info->add("fully connected info", fc_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fully_connected_inst::typed_primitive_inst(network_impl& network, fully_connected_node const& node)
    : parent(network, node) {
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input size",
                          input_layout.size.raw.size(),
                          "output size",
                          output_layout.size.raw.size(),
                          "");
}
}  // namespace cldnn
