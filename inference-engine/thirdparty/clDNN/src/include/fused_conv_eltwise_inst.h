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
#include "api_extension/fused_conv_eltwise.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<fused_conv_eltwise> : public typed_program_node_base<fused_conv_eltwise> {
    using parent = typed_program_node_base<fused_conv_eltwise>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog),
          split(this->get_primitive()->split()),
          depthwise_sep_opt(false),
          transposed(false) {
        if (get_primitive()->eltw.with_activation) {
            auto slope = get_primitive()->eltw.activation_negative_slope;
            if (slope == 0.f) {
                this->add_fused_activation(activation_func::relu, {0.0f, 0.0f});
            } else {
                this->add_fused_activation(activation_func::relu_negative_slope, { slope, 0.f });
            }
        }
    }

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    void set_depthwise_sep_opt(bool node_depthwise_sep_opt) { depthwise_sep_opt = node_depthwise_sep_opt; }
    bool get_depthwise_sep_opt() const { return depthwise_sep_opt; }

    void set_transposed(bool node_transposed) { transposed = node_transposed; }
    bool get_transposed() const { return transposed; }

    program_node& input(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= static_cast<int32_t>(desc->input.size()))
            throw std::range_error("input index too big");

        return get_dependency(idx);
    }

    program_node& weights(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(desc->input.size() + idx);
    }

    program_node& bias(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("bias offset too big");

        return get_dependency(desc->input.size() + this->get_split() + idx);
    }

    bool bias_term() const { return get_primitive()->conv.bias.size() > 0; }

private:
    int32_t split;
    bool depthwise_sep_opt;
    bool transposed;
};

using fused_conv_eltwise_node = typed_program_node<fused_conv_eltwise>;

template <>
class typed_primitive_inst<fused_conv_eltwise> : public typed_primitive_inst_base<fused_conv_eltwise> {
    using parent = typed_primitive_inst_base<fused_conv_eltwise>;

public:
    static layout calc_output_layout(fused_conv_eltwise_node const& node);
    static std::string to_string(fused_conv_eltwise_node const& node);

public:
    typed_primitive_inst(network_impl& network, fused_conv_eltwise_node const& node);

    memory_impl& weights_memory(size_t index) const {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("weights offset too big");

        return dep_memory(2 + index);
    }

    memory_impl& bias_memory(size_t index) const {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("bias offset too big");

        return dep_memory(2 + node.get_split() + index);
    }

    bool bias_term() const { return node.bias_term(); }
};

using fused_conv_eltwise_inst = typed_primitive_inst<fused_conv_eltwise>;

}  // namespace cldnn
