// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/lstm_dynamic.hpp"
#include "primitive_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<lstm_dynamic> : public typed_program_node_base<lstm_dynamic> {
    using parent = typed_program_node_base<lstm_dynamic>;

    typed_program_node(std::shared_ptr<primitive> prim, program& prog) : parent(prim, prog) {}

    program_node& input() const { return get_dependency(0); }
    float clip() const { return get_primitive()->clip; }
    bool input_forget() const { return get_primitive()->input_forget; }
    primitive_id bias_id() const { return get_primitive()->bias; }
    primitive_id weights_id() const { return get_primitive()->weights; }
    primitive_id recurrent_id() const { return get_primitive()->recurrent; }
    primitive_id initial_hidden_id() const { return get_primitive()->initial_hidden; }
    primitive_id initial_cell_id() const { return get_primitive()->initial_cell; }
    primitive_id dyn_length_id() const { return get_primitive()->dyn_length; }
    primitive_id last_hidden_state_id() const { return get_primitive()->last_hidden_state; }
    primitive_id last_cell_state_id() const { return get_primitive()->last_cell_state; }
};

using lstm_dynamic_node = typed_program_node<lstm_dynamic>;

template <>
class typed_primitive_inst<lstm_dynamic> : public typed_primitive_inst_base<lstm_dynamic> {
    using parent = typed_primitive_inst_base<lstm_dynamic>;
    using parent::parent;

public:
    static layout calc_output_layout(lstm_dynamic_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(lstm_dynamic_node const& node);

    typed_primitive_inst(network& network, lstm_dynamic_node const& node);

    static void check_direction(program_node& node, int32_t direction, std::string name) {
        if (node.get_output_layout().spatial(1) != direction)
            CLDNN_ERROR_MESSAGE(node.id(), name + " directions size need to equal 1 or 2 (bidrectional) !");
    }

    static void check_common_lstm_dynamic_sizes(program_node& node,
                                                int32_t batch_size,
                                                int32_t hidden_size,
                                                int32_t direction,
                                                std::string name) {
        auto node_layout = node.get_output_layout();
        CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                      name + " format",
                                      node.get_output_layout().format.value,
                                      "expected bfyx format",
                                      format::bfyx);
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              name + " batch size",
                              node_layout.batch(),
                              "input batch size",
                              batch_size,
                              "Sizes mismatch, " + name + ": " + node.id());
        check_direction(node, direction, name);
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              name + " x size",
                              node_layout.spatial(0),
                              "input_size",
                              hidden_size,
                              "Sizes mismatch, " + name + ": " + node.id());
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              name + " f size",
                              node_layout.feature(),
                              "1",
                              1,
                              "Sizes mismatch, " + name + ": " + node.id());
    }
};

using lstm_dynamic_inst = typed_primitive_inst<lstm_dynamic>;

}  // namespace cldnn
