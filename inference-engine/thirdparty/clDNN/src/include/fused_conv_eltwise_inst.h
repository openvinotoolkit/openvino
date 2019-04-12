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
#include "api_extension/CPP/fused_conv_eltwise.hpp"
#include "primitive_inst.h"

#include <memory>

namespace cldnn
{

template <>
struct typed_program_node<fused_conv_eltwise> : public typed_program_node_base<fused_conv_eltwise>
{
    using parent = typed_program_node_base<fused_conv_eltwise>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog)
        , split(this->get_primitive()->split())
        , depthwise_sep_opt(false)
        , transposed(false)
        , conv_input_qf(this->get_primitive()->conv.input_quantization_factor)
        , conv_output_qf(this->get_primitive()->conv.output_quantization_factor)
    {
    }

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    void set_depthwise_sep_opt(bool node_depthwise_sep_opt) { depthwise_sep_opt = node_depthwise_sep_opt; }
    bool get_depthwise_sep_opt() const { return depthwise_sep_opt; }

    void set_transposed(bool node_transposed) { transposed = node_transposed; }
    bool get_transposed() const { return transposed; }

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

        return get_dependency(desc->input.size() + 2 * this->get_split() + idx);
    }

    program_node& conv_output_calibration_factors(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("calibration factor offset too big");

        return get_dependency(desc->input.size() + 3 * this->get_split() + idx);
    }

    program_node& eltw_output_calibration_factors() const
    {
        return get_dependency(desc->input.size() + 4 * this->get_split());
    }

    bool bias_term() const
    {
        return get_primitive()->conv.bias.size() > 0;
    }

    bool weights_quantization_term() const
    {
        return get_primitive()->conv.weights_quantization_factors.size() > 0;
    }

    bool conv_output_calibration_term() const
    {
        return get_primitive()->conv.output_calibration_factors.size() > 0;
    }

    bool eltw_output_calibration_term() const
    {
        return get_primitive()->eltw.output_calibration_factors.size() > 0;
    }

    float get_conv_input_qf() const { return conv_input_qf; }
    float get_conv_output_qf() const { return conv_output_qf; }
    float get_eltw_output_qf() const { return eltw_output_qf; }

private:
    int32_t split;
    bool depthwise_sep_opt;
    bool transposed;
    float conv_input_qf;
    float conv_output_qf;
    float eltw_output_qf;
};

using fused_conv_eltwise_node = typed_program_node<fused_conv_eltwise>;

template <>
class typed_primitive_inst<fused_conv_eltwise> : public typed_primitive_inst_base<fused_conv_eltwise>
{
    using parent = typed_primitive_inst_base<fused_conv_eltwise>;

public:
    static layout calc_output_layout(fused_conv_eltwise_node const& node);
    static std::string to_string(fused_conv_eltwise_node const& node);

public:
    typed_primitive_inst(network_impl& network, fused_conv_eltwise_node const& node);

    memory_impl& weights_memory(size_t index) const
    {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("weights offset too big");
        
        return dep_memory(2 + index);
    }

    memory_impl& bias_memory(size_t index) const
    { 
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("bias offset too big");

        return dep_memory(2 + node.get_split() + index);
    }

    memory_impl& weights_quantization_factors_memory(size_t index) const
    {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("quantization factors offset too big");

        return dep_memory(2 + 2*node.get_split() + index);
    }

    memory_impl& output_calibration_factors_memory(size_t index) const
    {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("quantization factors offset too big");

        return dep_memory(2 + 3 * node.get_split() + index);
    }

    memory_impl& eltw_output_calibration_factors_memory() const
    {
        return dep_memory(2 + 4 * node.get_split());
    }

    bool bias_term() const
    {
        return node.bias_term();
    }

    bool weights_quantization_factors_term() const
    {
        return node.weights_quantization_term();
    }

    bool conv_output_calibration_factors_term() const
    {
        return node.conv_output_calibration_term();
    }

    bool eltw_output_calibration_factors_term() const
    {
        return node.eltw_output_calibration_term();
    }
};

using fused_conv_eltwise_inst = typed_primitive_inst<fused_conv_eltwise>;

}
