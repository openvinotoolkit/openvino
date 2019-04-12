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
#pragma once
#include "api_extension/CPP/fused_conv_bn_scale.hpp"
#include "primitive_inst.h"

#include <memory>

namespace cldnn
{

template <>
struct typed_program_node<fused_conv_bn_scale> : public typed_program_node_base<fused_conv_bn_scale>
{
    using parent = typed_program_node_base<fused_conv_bn_scale>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog)
        , split(this->get_primitive()->split())
    {
    }

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    program_node& input(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= static_cast<int32_t>(desc->input.size()))
            throw std::range_error("input index too big");

        return get_dependency(idx);
    }

    program_node& weights(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(desc->input.size() + idx);
    }

    program_node& bias(size_t idx = 0) const
    { 
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("bias offset too big");

        return get_dependency(desc->input.size() + this->get_split() + idx);
    }

    program_node& weights_quantization_factors(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("quantization factor offset too big");

        return get_dependency(desc->input.size() + 2*this->get_split() + idx);
    }

    program_node& output_calibration_factors(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("calibration factor offset too big");

        return get_dependency(desc->input.size() + 3 * this->get_split() + idx);
    }

    bool bias_term() const
    {
        return get_primitive()->bias.size() > 0;
    }

    bool scale_bias_term() const
    {
        return !get_primitive()->scale_bias.empty();
    }

    bool is_fused_in_training() const
    {
        return !get_primitive()->inv_variance.empty();
    }

private:
    int32_t split;
};

using fused_conv_bn_scale_node = typed_program_node<fused_conv_bn_scale>;

template <>
class typed_primitive_inst<fused_conv_bn_scale> : public typed_primitive_inst_base<fused_conv_bn_scale>
{
    using parent = typed_primitive_inst_base<fused_conv_bn_scale>;

public:
    static layout calc_output_layout(fused_conv_bn_scale_node const& node);
    static std::string to_string(fused_conv_bn_scale_node const& node);

public:
    typed_primitive_inst(network_impl& network, fused_conv_bn_scale_node const& node);

    memory_impl& weights_memory(size_t index) const
    {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("weights offset too big");
        
        return dep_memory(inputs_memory_count() + index);
    }

    memory_impl& bias_memory(size_t index) const
    { 
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("bias offset too big");

        return dep_memory(inputs_memory_count() + node.get_split() + index);
    }

    bool bias_term() const
    {
        return node.bias_term();
    }

    bool scale_bias_term() const
    {
        return node.scale_bias_term();
    }

    bool is_fused_in_training() const
    {
        return node.is_fused_in_training();
    }
};

using fused_conv_bn_scale_inst = typed_primitive_inst<fused_conv_bn_scale>;

}
