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
#include "api/CPP/convolution.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<convolution> : public typed_program_node_base<convolution> {
    using parent = typed_program_node_base<convolution>;

public:
    struct fused_primitive_desc {
        std::shared_ptr<const primitive> prim;
        size_t dep_start_idx;
        std::vector<primitive_id> deps;
        cldnn_activation_func_t activation;
        cldnn_activation_additional_params activation_params;
    };


    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog),
          split(this->get_primitive()->split()),
          depthwise_sep_opt(false),
          transposed(false),
          input_qf(this->get_primitive()->input_quantization_factor),
          output_qf(this->get_primitive()->output_quantization_factor),
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

    program_node& weights_quantization_factors(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("quantization factor offset too big");


        return get_dependency(1 + (1 + 1 * bias_term()) * this->get_split() + idx + get_trans_dep_offset());
    }

    program_node& trans() const {
        if (!deformable_mode)
            throw std::range_error("trans input exists only in deformable mode");

        return get_dependency(1);
    }

    program_node& output_calibration_factors(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("calibration factor offset too big");

        return get_dependency(1 + (1 + 1 * bias_term() + 1 * weights_quantization_term()) * this->get_split() + idx +
                              get_trans_dep_offset());
    }

    program_node& fused_eltwise(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("eltwise offset too big");

        int index = 1 + this->get_split()
                    + (bias_term() ? this->get_split() : 0)
                    + (weights_quantization_term() ? this->get_split() : 0)
                    + (output_calibration_term() ? this->get_split() : 0);
        return get_dependency(static_cast<size_t>(index));
    }

    void add_fused_primitive(const program_node *p) {
        fused_primitive_desc local_desc;
        local_desc.prim = p->get_primitive();
        local_desc.dep_start_idx = this->get_dependencies().size();
        local_desc.activation = cldnn_activation_func_t::activation_none;
        if (p->get_fused_activation_func() != cldnn_activation_func_t::activation_none) {
            local_desc.activation = p->get_fused_activation_func();
            local_desc.activation_params = p->get_fused_activation_params();
        }

        for (size_t i = 0; i < p->get_dependencies().size(); i++) {
            auto& dep = p->get_dependency(i);
            if (dep.id() == this->id())
                continue;

            this->dependencies.push_back(&dep);
            local_desc.deps.push_back(dep.id());
            dep.users.push_back(this);
        }
        fused_prims.push_back(local_desc);
    }

    const std::vector<fused_primitive_desc>& get_fused_primitives() const {
        return fused_prims;
    }

    bool bias_term() const { return get_primitive()->bias.size() > 0; }

    bool weights_quantization_term() const { return get_primitive()->weights_quantization_factors.size() > 0; }

    bool output_calibration_term() const { return get_primitive()->output_calibration_factors.size() > 0; }

    size_t get_fused_inputs_count() const {
        size_t count = 0;
        for (auto& fp : get_fused_primitives()) {
            count += fp.deps.size();
        }
        return count;
    }

    float get_input_qf() const { return input_qf; }
    float get_output_qf() const { return output_qf; }

private:
    int32_t split;
    bool depthwise_sep_opt;
    bool transposed;
    float input_qf;
    float output_qf;
    uint32_t groups;
    uint32_t deformable_groups;
    bool deformable_mode;
    std::vector<fused_primitive_desc> fused_prims;
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

    memory_impl& weights_quantization_factors_memory(size_t index) const {
        if (node.get_groups() == 1) {
            if (static_cast<int32_t>(index) >= node.get_split())
                throw std::range_error("quantization factors offset too big");
            return dep_memory(1 + (1 + 1 * bias_term()) * node.get_split() + index + node.get_trans_dep_offset());
        } else {  // all quantization_factors are in one buffer
            return dep_memory(2 + 1 * bias_term() + node.get_trans_dep_offset());
        }
    }

    memory_impl& trans_memory() const {
        if (!node.get_trans_dep_offset())
            throw std::range_error("trans input exists only in deformable mode");
        return dep_memory(1);
    }

    memory_impl& output_calibration_factors_memory(size_t index) const {
        if (node.get_groups() == 1) {
            if (static_cast<int32_t>(index) >= node.get_split())
                throw std::range_error("quantization factors offset too big");
            return dep_memory(1 + (1 + 1 * bias_term() + 1 * weights_quantization_factors_term()) * node.get_split() +
                              index + node.get_trans_dep_offset());
        } else {  // all calibration_factors are in one buffer
            return dep_memory(2 + 1 * bias_term() + 1 * weights_quantization_factors_term() +
                              node.get_trans_dep_offset());
        }
    }

    memory_impl& fused_memory(size_t dep_id) const {
        int index = 1 + node.get_split()
                    + (bias_term() ? node.get_split() : 0)
                    + (weights_quantization_factors_term() ? node.get_split() : 0)
                    + (output_calibration_factors_term() ? node.get_split() : 0);
        return dep_memory(index + dep_id);
    }

    bool bias_term() const { return node.bias_term(); }

    bool weights_quantization_factors_term() const { return node.weights_quantization_term(); }

    bool output_calibration_factors_term() const { return node.output_calibration_term(); }

    bool has_fused_primitives() const { return !node.get_fused_primitives().empty(); }

    size_t get_fused_mem_count() const { return node.get_fused_inputs_count(); }
};

using convolution_inst = typed_primitive_inst<convolution>;

}  // namespace cldnn
