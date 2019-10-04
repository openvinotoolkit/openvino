/*
// Copyright (c) 2016-2018 Intel Corporation
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
#include "api/convolution.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<deformable_conv> : public typed_program_node_base<deformable_conv> {
    using parent = typed_program_node_base<deformable_conv>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
            : parent(prim, prog),
              split(this->get_primitive()->split()),
              depthwise_sep_opt(false),
              groups(this->get_primitive()->groups) {
        support_padding_all(true);
    }

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    void set_depthwise_sep_opt(bool node_depthwise_sep_opt) { depthwise_sep_opt = node_depthwise_sep_opt; }
    bool get_depthwise_sep_opt() const { return depthwise_sep_opt; }

    void set_transposed(bool node_transposed) { transposed = node_transposed; }
    bool get_transposed() const { return transposed; }

    void set_groups(uint32_t node_groups) { groups = node_groups; }
    uint32_t get_groups() const { return groups; }

    program_node& input() const { return get_dependency(0); }

    program_node& weights(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(1 + idx);
    }

    program_node& bias(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("bias offset too big");

        return get_dependency(1 + this->get_split() + idx);
    }

    bool bias_term() const { return get_primitive()->bias.size() > 0; }

private:
    int32_t split;
    bool depthwise_sep_opt;
    bool transposed;
    uint32_t groups;
};

using deformable_conv_node = typed_program_node<deformable_conv>;

template <>
class typed_primitive_inst<deformable_conv> : public typed_primitive_inst_base<deformable_conv> {
    using parent = typed_primitive_inst_base<deformable_conv>;

public:
    static layout calc_output_layout(deformable_conv_node const& node);
    static std::string to_string(deformable_conv_node const& node);

public:
    typed_primitive_inst(network_impl& network, deformable_conv_node const& node);

    memory_impl& weights_memory(size_t index) const {
        if (node.get_groups() == 1) {
            if (static_cast<int32_t>(index) >= node.get_split())
                throw std::range_error("weights offset too big");
            return dep_memory(1 + index);
        } else {  // all weights are in one buffer
            return dep_memory(1);
        }
    }

    memory_impl& bias_memory(size_t index) const {
        if (node.get_groups() == 1) {
            if (static_cast<int32_t>(index) >= node.get_split())
                throw std::range_error("bias offset too big");
            return dep_memory(1 + node.get_split());
        } else {  // all bias are in one buffer
            return dep_memory(2);
        }
    }

    bool bias_term() const { return node.bias_term(); }
};

using deformable_conv_inst = typed_primitive_inst<deformable_conv>;

template <>
struct typed_program_node<deformable_interp> : public typed_program_node_base<deformable_interp> {
    using parent = typed_program_node_base<deformable_interp>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
            : parent(prim, prog),
              split(1),
              depthwise_sep_opt(false),
              transposed(false),
              groups(this->get_primitive()->groups),
              deformable_groups(this->get_primitive()->deformable_groups) {
        support_padding_all(true);
    }

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    void set_depthwise_sep_opt(bool node_depthwise_sep_opt) { depthwise_sep_opt = node_depthwise_sep_opt; }
    bool get_depthwise_sep_opt() const { return depthwise_sep_opt; }

    void set_transposed(bool node_transposed) { transposed = node_transposed; }
    bool get_transposed() const { return transposed; }

    void set_groups(uint32_t node_groups) { groups = node_groups; }
    uint32_t get_groups() const { return groups; }

    void set_deformable_groups(uint32_t node_deformable_groups) { deformable_groups = node_deformable_groups; }
    uint32_t get_deformable_groups() const { return deformable_groups; }

    program_node& input() const { return get_dependency(0); }
    program_node& trans() const { return get_dependency(1); }

private:
    int32_t split;
    bool depthwise_sep_opt;
    bool transposed;
    uint32_t groups;
    uint32_t deformable_groups;
};

using deformable_interp_node = typed_program_node<deformable_interp>;

template <>
class typed_primitive_inst<deformable_interp> : public typed_primitive_inst_base<deformable_interp> {
    using parent = typed_primitive_inst_base<deformable_interp>;

public:
    static layout calc_output_layout(deformable_interp_node const& node);
    static std::string to_string(deformable_interp_node const& node);

public:
    typed_primitive_inst(network_impl& network, deformable_interp_node const& node);

    memory_impl& trans_memory() const { return dep_memory(1); }
};

using deformable_interp_inst = typed_primitive_inst<deformable_interp>;

}  // namespace cldnn
