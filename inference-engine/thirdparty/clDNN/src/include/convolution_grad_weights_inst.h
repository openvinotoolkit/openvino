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
#include "api/CPP/convolution_grad_weights.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<convolution_grad_weights> : public typed_program_node_base<convolution_grad_weights>
{
    using parent = typed_program_node_base<convolution_grad_weights>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog)
        , split(this->get_primitive()->split())
        , depthwise_sep_opt(false)
    {
    }

    
    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    void set_depthwise_sep_opt(bool node_depthwise_sep_opt) { depthwise_sep_opt = node_depthwise_sep_opt; }
    bool get_depthwise_sep_opt() const { return depthwise_sep_opt; }

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }

    program_node& weights(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(2 + idx);
    }

    program_node& bias(size_t idx = 0) const
    { 
        if (static_cast<int32_t>(idx) >= get_split())
            throw std::range_error("bias offset too big");

        return get_dependency(2 + this->get_split() + idx);
    }

    program_node& prev_weights_grad(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= get_split())
            throw std::range_error("prev weights grad offset too big");
        return get_dependency(2 + (bias_term() ? 2 : 1) * get_split() + idx);
    }

    program_node& prev_bias_grad(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= get_split())
            throw std::range_error("prev bias grad offset too big");
        return get_dependency(2 + 3 * get_split() + idx);
    }

    bool use_momentum() const
    {
        if (get_primitive()->prev_weights_grad.size() != 0)
            return true;
        else
            return false;
    }

    bool bias_term() const
    {
        if (get_primitive()->bias.size() != 0)
            return true;
        else
            return false;
    }

private:
    int32_t split;
    bool depthwise_sep_opt;
};

using convolution_grad_weights_node = typed_program_node<convolution_grad_weights>;

template <>
class typed_primitive_inst<convolution_grad_weights> : public typed_primitive_inst_base<convolution_grad_weights>
{
    using parent = typed_primitive_inst_base<convolution_grad_weights>;

public:
    static layout calc_output_layout(convolution_grad_weights_node const& node);
    static std::string to_string(convolution_grad_weights_node const& node);

public:
    typed_primitive_inst(network_impl& network, convolution_grad_weights_node const& node);

    memory_impl& weights_memory(size_t index) const
    {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("weights offset too big");

        return dep_memory(2 + index);
    }

    memory_impl& bias_memory(size_t index) const
    {
        if (argument.bias.size() == 0 && static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("no bias data");

        if (static_cast<int32_t>(index) > node.get_split())
            throw std::range_error("bias offset too big");

        return dep_memory(2 + node.get_split() + index);
    }

    memory_impl& prev_weights_grad(size_t index) const
    {
        if(argument.prev_weights_grad.size() == 0 && static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("no prev weights grad data");

        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("prev weights grad offset too big");

        return dep_memory(2 + (bias_term() ? 2 : 1) * node.get_split() + index);
    }

    memory_impl& prev_bias_grad(size_t index) const
    {
        if (argument.prev_bias_grad.size() == 0 && static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("no prev bias grad data");

        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("prev bias grad offset too big");

        return dep_memory(2 + 3 * node.get_split() + index);
    }

    bool use_momentum() const
    {
        if (argument.prev_weights_grad.size() != 0)
            return true;
        else
            return false;
    }

    bool bias_term() const
    {
        if (argument.bias.size() != 0)
            return true;
        else
            return false;
    }
};

using convolution_grad_weights_inst = typed_primitive_inst<convolution_grad_weights>;

}
