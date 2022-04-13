// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/lstm_dynamic_input.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<lstm_dynamic_input> : public typed_program_node_base<lstm_dynamic_input> {
    using parent = typed_program_node_base<lstm_dynamic_input>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog) : parent(prim, prog) {}

    program_node& input() const { return *get_dependency(0).first; }
    program_node& dyn_length() const { return *get_dependency(1).first; }
    program_node& weights() const { return *get_dependency(2).first; }

    program_node& bias() const {
        CLDNN_ERROR_BOOL(id(), "Bias term", !bias_term(), "Trying to get non existing bias.");
        return *get_dependency(3).first;
    }

    int32_t direction() const { return weights().get_output_layout().size.feature[0]; }
    bool dyn_length_term() const { return !get_primitive()->dyn_length.empty(); }
    bool bias_term() const { return !get_primitive()->bias.empty(); }
    bool weights_term() const { return !get_primitive()->weights.empty(); }
};

using lstm_dynamic_input_node = typed_program_node<lstm_dynamic_input>;

template <>
class typed_primitive_inst<lstm_dynamic_input> : public typed_primitive_inst_base<lstm_dynamic_input> {
    using parent = typed_primitive_inst_base<lstm_dynamic_input>;

public:
    static layout calc_output_layout(lstm_dynamic_input_node const& node);
    static std::string to_string(lstm_dynamic_input_node const& node);

public:
    typed_primitive_inst(network& network, lstm_dynamic_input_node const& node);

    memory::ptr dyn_length_memory() const { return dep_memory_ptr(1); }
    memory::ptr weights_memory() const { return dep_memory_ptr(2); }
    memory::ptr bias_memory() const {
        CLDNN_ERROR_BOOL(id(), "Bias term", !bias_term(), "Trying to get non existing bias memory.");
        return dep_memory_ptr(3);
    }
    int32_t direction() const { return node.direction(); }
    bool bias_term() const { return node.bias_term(); }
};

using lstm_dynamic_input_inst = typed_primitive_inst<lstm_dynamic_input>;

}  // namespace cldnn
