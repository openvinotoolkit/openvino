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

#include "gather_tree_inst.h"

#include "error_handler.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <algorithm>

namespace cldnn {
primitive_type_id gather_tree::type_id() {
    static primitive_type_base<gather_tree> instance;
    return &instance;
}

layout gather_tree_inst::calc_output_layout(gather_tree_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
        "Output data type forcing is not supported for gather_tree_node!");
    auto input_layout = node.input().get_output_layout();
    return input_layout;
}

std::string gather_tree_inst::to_string(gather_tree_node const& node) {
    std::stringstream primitive_description;
    node.desc_to_json()->dump(primitive_description);
    return primitive_description.str();
}

gather_tree_inst::typed_primitive_inst(network_impl& network, gather_tree_node const& node) : parent(network, node) {
    auto input_layout = node.input().get_output_layout();

    const auto input_format = input_layout.format;

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
        "Input format",
        input_format.value,
        "supported border primitive input formats",
        format::bfyx,
        format::yxfb,
        format::byxf);

    auto dependencies = node.get_dependencies();

    // check input dims
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input0 size", dependencies.at(0)->get_output_layout().size, "output size", input_layout.size,
        "mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input1 size", dependencies.at(1)->get_output_layout().size, "output size", input_layout.size,
        "mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input2 size", dependencies.at(2)->get_output_layout().count(), "node's feature size", input_layout.size.feature.at(0),
        "There can't be more than one end_token");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input3 size", dependencies.at(3)->get_output_layout().size.count(), "one", 1,
        "There can't be more than one end_token");
}
}  // namespace cldnn
