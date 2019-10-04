/*
// Copyright (c) 2016 Intel Corporation
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
#pragma once
#include "api/fully_connected.hpp"
#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<fully_connected> : public typed_program_node_base<fully_connected> {
    using parent = typed_program_node_base<fully_connected>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog),
          input_qf(this->get_primitive()->input_quantization_factor),
          output_qf(this->get_primitive()->output_quantization_factor) {}

    program_node& input() const { return get_dependency(0); }
    program_node& weights() const { return get_dependency(1); }
    program_node& bias() const { return get_dependency(2); }
    program_node& weights_quantization_factors() const { return get_dependency(3); }
    program_node& output_calibration_factors() const { return get_dependency(4); }
    bool bias_term() const { return !get_primitive()->bias.empty(); }
    bool weights_quantization_term() const { return !get_primitive()->weights_quantization_factors.empty(); }
    bool output_calibration_term() const { return !get_primitive()->output_calibration_factors.empty(); }
    float get_input_qf() const { return input_qf; }
    float get_output_qf() const { return output_qf; }

private:
    float input_qf;
    float output_qf;
};

using fully_connected_node = typed_program_node<fully_connected>;

template <>
class typed_primitive_inst<fully_connected> : public typed_primitive_inst_base<fully_connected> {
    using parent = typed_primitive_inst_base<fully_connected>;

public:
    static layout calc_output_layout(fully_connected_node const& node);
    static std::string to_string(fully_connected_node const& node);

public:
    typed_primitive_inst(network_impl& network, fully_connected_node const& node);

    memory_impl& weights_memory() const { return dep_memory(1); }
    memory_impl& bias_memory() const { return dep_memory(2); }
    memory_impl& weights_quantization_factors_memory() const { return dep_memory(3); }
    memory_impl& output_calibration_factors_memory() const { return dep_memory(4); }

    bool bias_term() const { return !argument.bias.empty(); }
    bool weights_quantization_factors_term() const { return node.weights_quantization_term(); }
    bool output_calibration_factors_term() const { return node.output_calibration_term(); }
};

using fully_connected_inst = typed_primitive_inst<fully_connected>;

}  // namespace cldnn
