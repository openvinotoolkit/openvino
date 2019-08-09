/*
// Copyright (c) 2018 Intel Corporation
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
#include "select_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id select_type_id() {
    static primitive_type_base<select> instance;
    return &instance;
}

layout select_inst::calc_output_layout(select_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for select_node!");
    return node.input().get_non_padded_output_layout();
}

std::string select_inst::to_string(select_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite select_info;
    for (size_t i = 0; i < node.inputs_count(); i++) {
        select_info.add("input_" + std::to_string(i), node.input(i).id());
    }

    node_info->add("select info", select_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

select_inst::typed_primitive_inst(network_impl& network, select_node const& node) : parent(network, node) {
    auto& deps = node.get_dependencies();

    for (size_t i = 0; i < deps.size() - 1; i++) {
        auto batch1 = deps[i]->get_output_layout().size.batch[0];
        auto batch2 = deps[i + 1]->get_output_layout().size.batch[0];
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Batch size input", batch1, "Batch size next input", batch2, "");

        auto feature1 = deps[i]->get_output_layout().size.feature[0];
        auto feature2 = deps[i + 1]->get_output_layout().size.feature[0];
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Feature size input", feature1, "Feature size next input", feature2, "");

        auto spatial1 = deps[i]->get_output_layout().size.spatial[0];
        auto spatial2 = deps[i + 1]->get_output_layout().size.spatial[0];
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Spatial size input", spatial1, "Spatial size next input", spatial2, "");

        auto format1 = deps[i]->get_output_layout().format;
        auto format2 = deps[i + 1]->get_output_layout().format;
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Format input", format1, "Format next input", format2, "");
    }

    auto data_type1 = deps[0]->get_output_layout().data_type;
    auto data_type2 = deps[1]->get_output_layout().data_type;
    CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(), "Data type input 1", data_type1, "Data type input 2", data_type2, "");
}
}  // namespace cldnn
