// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/lstm.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<lstm_gemm> : public typed_program_node_base<lstm_gemm> {
    using parent = typed_program_node_base<lstm_gemm>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& weights() const { return get_dependency(1); }
    program_node& recurrent() const { return get_dependency(2); }
    program_node& bias() const { return get_dependency(3); }
    program_node& hidden() const { return bias_term() ? get_dependency(4) : get_dependency(3); }
    bool bias_term() const { return !get_primitive()->bias.empty(); }
    bool hidden_term() const { return !get_primitive()->hidden.empty(); }
    uint32_t direction() const { return get_primitive()->direction; }
};

using lstm_gemm_node = typed_program_node<lstm_gemm>;

template <>
class typed_primitive_inst<lstm_gemm> : public typed_primitive_inst_base<lstm_gemm> {
    using parent = typed_primitive_inst_base<lstm_gemm>;

public:
    static layout calc_output_layout(lstm_gemm_node const& node);
    static std::string to_string(lstm_gemm_node const& node);

public:
    typed_primitive_inst(network& network, lstm_gemm_node const& node);

    memory::ptr weights_memory() const { return dep_memory_ptr(1); }
    memory::ptr recurrent_memory() const { return dep_memory_ptr(2); }
    memory::ptr bias_memory() const { return dep_memory_ptr(3); }
    memory::ptr hidden_memory() const { return bias_term() ? dep_memory_ptr(4) : dep_memory_ptr(3); }
    bool bias_term() const { return !argument.bias.empty(); }
    bool hidden_term() const { return !argument.hidden.empty(); }
    uint32_t direction() const { return argument.direction; }
};

using lstm_gemm_inst = typed_primitive_inst<lstm_gemm>;

}  // namespace cldnn
