// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "condition_inst.h"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>

namespace cldnn {
primitive_type_id condition::type_id() {
    static primitive_type_base<condition> instance;
    return &instance;
}
/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout condition_inst::calc_output_layout(condition_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_types.at(0)) == false &&
           "Output data type forcing is not supported for condition_node!");
    node.set_branches();

    auto branch_true_output = node.get_branch_true()->get_outputs();
    auto branch_false_output = node.get_branch_false()->get_outputs();
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Count of branch true outputs",
                          branch_true_output.size(),
                          "expected outputs size",
                          1,
                          "Branch true should have one output.");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Count of branch false outputs",
                          branch_false_output.size(),
                          "expected outputs size",
                          1,
                          "Branch false should have one output.");

    auto layout_true = branch_true_output.at(0)->get_output_layout();
    auto layout_false = branch_false_output.at(0)->get_output_layout();
    CLDNN_ERROR_LAYOUT_MISMATCH(node.id(),
                                "Branch true output layout",
                                layout_true,
                                "branch false output layout",
                                layout_false,
                                "Layout of the branches should be the same.");

    return layout_true;
}

std::string condition_inst::to_string(condition_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite condition_info;

    node_info->add("condition info", condition_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
Condition primitive is resuing memory with the input.
*/
condition_inst::typed_primitive_inst(network& network, condition_node const& node)
    : parent(network, node),
      _net_true(network::allocate_network(node.get_program().get_engine(), node.get_branch_true(), true)),
      _net_false(network::allocate_network(node.get_program().get_engine(), node.get_branch_false(), true)) {
    auto compare_tensor = node.compare().get_output_layout().size;
    auto input_tensor = node.input().get_output_layout().size;
    CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                          "Compare tensor",
                                          compare_tensor,
                                          "input tensor",
                                          input_tensor,
                                          "Compare primitive is too big.");

    auto compare_with_offster_tensor = compare_tensor + node.offset();
    CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                          "Offset with compare tensor",
                                          compare_with_offster_tensor,
                                          "input tensor",
                                          input_tensor,
                                          "Offset is too big.");
}
}  // namespace cldnn
