// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/lstm_dynamic.hpp"
#include "primitive_inst.h"
#include "error_handler.h"
#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<lstm_dynamic> : public typed_program_node_base<lstm_dynamic> {
    using parent = typed_program_node_base<lstm_dynamic>;

    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog) : parent(prim, prog) {}

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

public:
    static layout calc_output_layout(lstm_dynamic_node const& node);
    static std::string to_string(lstm_dynamic_node const& node);

    typed_primitive_inst(network_impl& network, lstm_dynamic_node const& node);

    static void check_direction(program_node& node, int32_t direction, std::string name) {
        if (node.get_output_layout().size.spatial[1] != direction)
            CLDNN_ERROR_MESSAGE(node.id(), name + " directions size need to equal 1 or 2 (bidrectional) !");
    }

    static void check_common_lstm_dynamic_sizes(program_node& node,
                                                int32_t batch_size,
                                                int32_t hidden_size,
                                                int32_t direction,
                                                std::string name) {
        auto node_tensor = node.get_output_layout().size;
        CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                      name + " format",
                                      node.get_output_layout().format.value,
                                      "expected bfyx format",
                                      format::bfyx);
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              name + " batch size",
                              node_tensor.batch[0],
                              "input batch size",
                              batch_size,
                              "Sizes mismatch, " + name + ": " + node.id());
        check_direction(node, direction, name);
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              name + " x size",
                              node_tensor.spatial[0],
                              "input_size",
                              hidden_size,
                              "Sizes mismatch, " + name + ": " + node.id());
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              name + " f size",
                              node_tensor.feature[0],
                              "1",
                              1,
                              "Sizes mismatch, " + name + ": " + node.id());
    }
};

using lstm_dynamic_inst = typed_primitive_inst<lstm_dynamic>;

}  // namespace cldnn
