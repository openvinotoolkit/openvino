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
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<convolution> : public typed_program_node_base<convolution> {
    using parent = typed_program_node_base<convolution>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog),
          split(this->get_primitive()->split()),
          depthwise_sep_opt(false),
          transposed(false),
          groups(this->get_primitive()->groups),
          deformable_groups(this->get_primitive()->deformable_groups),
          deformable_mode(this->get_primitive()->deformable_mode) {
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

    int32_t get_trans_dep_offset() const { return deformable_mode ? 1 : 0; }

    program_node& input() const { return get_dependency(0); }

    program_node& weights(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(1 + idx + get_trans_dep_offset());
    }

    program_node& bias(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("bias offset too big");

        return get_dependency(1 + this->get_split() + idx + get_trans_dep_offset());
    }

    program_node& weights_zero_points(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights zero points offset too big");

        return get_dependency(1 + (1 + 1 * bias_term()) * this->get_split() + idx + get_trans_dep_offset());
    }

    program_node& activations_zero_points(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("activations zero points offset too big");

        return get_dependency(1 + (1 + 1 * bias_term() + 1 * weights_zero_points_term()) * this->get_split() + idx +
                              get_trans_dep_offset());
    }

    program_node& compensation(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("activations zero points offset too big");

        return get_dependency(1 + (1 + 1 * bias_term() + 1 * weights_zero_points_term() + 1*activations_zero_points_term()) * this->get_split()
                              + idx + get_trans_dep_offset());
    }

    program_node& trans() const {
        if (!deformable_mode)
            throw std::range_error("trans input exists only in deformable mode");

        return get_dependency(1);
    }

    bool bias_term() const { return get_primitive()->bias.size() > 0; }

    bool weights_zero_points_term() const { return get_primitive()->weights_zero_points.size() > 0; }
    bool compensation_term() const { return get_primitive()->compensation.size() > 0; }
    bool activations_zero_points_term() const { return get_primitive()->activations_zero_points.size() > 0; }

private:
    int32_t split;
    bool depthwise_sep_opt;
    bool transposed;
    uint32_t groups;
    uint32_t deformable_groups;
    bool deformable_mode;
};

using convolution_node = typed_program_node<convolution>;

template <>
class typed_primitive_inst<convolution> : public typed_primitive_inst_base<convolution> {
    using parent = typed_primitive_inst_base<convolution>;

public:
    static layout calc_output_layout(convolution_node const& node);
    static std::string to_string(convolution_node const& node);

public:
    typed_primitive_inst(network_impl& network, convolution_node const& node);

    memory_impl& weights_memory(size_t index) const {
        if (node.get_groups() == 1) {
            if (static_cast<int32_t>(index) >= node.get_split())
                throw std::range_error("weights offset too big");
            return dep_memory(1 + index + node.get_trans_dep_offset());
        } else {  // all weights are in one buffer
            return dep_memory(1 + node.get_trans_dep_offset());
        }
    }

    memory_impl& bias_memory(size_t index) const {
        if (node.get_groups() == 1) {
            if (static_cast<int32_t>(index) >= node.get_split())
                throw std::range_error("bias offset too big");
            return dep_memory(1 + node.get_split() + index + node.get_trans_dep_offset());
        } else {  // all bias are in one buffer
            return dep_memory(2 + node.get_trans_dep_offset());
        }
    }

    memory_impl& weights_zero_points_memory(size_t) const {
        if (node.get_split() > 1)
            throw std::range_error("Split is unsupported for quantized convolutions");
        return dep_memory(2 + 1 * bias_term() + node.get_trans_dep_offset());
    }

    memory_impl& trans_memory() const {
        if (!node.get_trans_dep_offset())
            throw std::range_error("trans input exists only in deformable mode");
        return dep_memory(1);
    }

    memory_impl& activations_zero_points_memory(size_t) const {
        if (node.get_split() > 1)
            throw std::range_error("Split is unsupported for quantized convolutions");
        return dep_memory(2 + 1 * bias_term() + 1 * weights_zero_points_term() + node.get_trans_dep_offset());
    }

    memory_impl& compensation_memory(size_t) const {
        if (node.get_split() > 1)
            throw std::range_error("Split is unsupported for quantized convolutions");
        return dep_memory(2 + 1 * bias_term() + 1 * weights_zero_points_term() + 1*activations_zero_points_term() + node.get_trans_dep_offset());
    }

    bool bias_term() const { return node.bias_term(); }

    bool weights_zero_points_term() const { return node.weights_zero_points_term(); }
    bool compensation_term() const { return node.compensation_term(); }
    bool activations_zero_points_term() const { return node.activations_zero_points_term(); }
};

using convolution_inst = typed_primitive_inst<convolution>;

}  // namespace cldnn
