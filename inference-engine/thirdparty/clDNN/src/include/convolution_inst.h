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
#include "api/CPP/convolution.hpp"
#include "primitive_inst.h"

#include <memory>

namespace cldnn
{

template <>
struct typed_program_node<convolution> : public typed_program_node_base<convolution>
{
    using parent = typed_program_node_base<convolution>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog)
        , split(this->get_primitive()->split())
        , depthwise_sep_opt(false)
        , transposed(false)
        , input_qf(this->get_primitive()->input_quantization_factor)
        , output_qf(this->get_primitive()->output_quantization_factor)
    {
    }

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    void set_depthwise_sep_opt(bool node_depthwise_sep_opt) { depthwise_sep_opt = node_depthwise_sep_opt; }
    bool get_depthwise_sep_opt() const { return depthwise_sep_opt; }

    void set_transposed(bool node_transposed) { transposed = node_transposed; }
    bool get_transposed() const { return transposed; }

    decltype(auto) input() const { return get_dependency(0); }

    decltype(auto) weights(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(1 + idx);
    }

    decltype(auto) bias(size_t idx = 0) const
    { 
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("bias offset too big");

        return get_dependency(1 + this->get_split() + idx);
    }

    decltype(auto) weights_quantization_factors(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("quantization factor offset too big");

        return get_dependency(1 + 2*this->get_split() + idx);
    }

    decltype(auto) output_calibration_factors(size_t idx = 0) const
    {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("calibration factor offset too big");

        return get_dependency(1 + 3 * this->get_split() + idx);
    }

    bool bias_term() const
    {
        return get_primitive()->bias.size() > 0;
    }

    bool weights_quantization_term() const
    {
        return get_primitive()->weights_quantization_factors.size() > 0;
    }

    bool output_calibration_term() const
    {
        return get_primitive()->output_calibration_factors.size() > 0;
    }
    
    float get_input_qf() const { return input_qf; }
    float get_output_qf() const { return output_qf; }

private:
    int32_t split;
    bool depthwise_sep_opt;
    bool transposed;
    float input_qf;
    float output_qf;
};

using convolution_node = typed_program_node<convolution>;

template <>
class typed_primitive_inst<convolution> : public typed_primitive_inst_base<convolution>
{
    using parent = typed_primitive_inst_base<convolution>;

public:
    static layout calc_output_layout(convolution_node const& node);
    static std::string to_string(convolution_node const& node);

public:
    typed_primitive_inst(network_impl& network, convolution_node const& node);

    decltype(auto)weights_memory(size_t index) const
    {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("weights offset too big");
        
        return dep_memory(1 + index);
    }

    decltype(auto) bias_memory(size_t index) const
    { 
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("bias offset too big");

        return dep_memory(1 + node.get_split() + index);
    }

    decltype(auto) weights_quantization_factors_memory(size_t index) const
    {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("quantization factors offset too big");

        return dep_memory(1 + 2*node.get_split() + index);
    }

    decltype(auto) output_calibration_factors_memory(size_t index) const
    {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("quantization factors offset too big");

        return dep_memory(1 + 3 * node.get_split() + index);
    }

    bool bias_term() const
    {
        return node.bias_term();
    }

    bool weights_quantization_factors_term() const
    {
        return node.weights_quantization_term();
    }

    bool output_calibration_factors_term() const
    {
        return node.output_calibration_term();
    }
};

using convolution_inst = typed_primitive_inst<convolution>;

}
