/*
// Copyright (c) 2016-2019 Intel Corporation
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
#include "api/eltwise.hpp"
#include "primitive_inst.h"
#include <memory>
#include "topology_impl.h"
#include "kernel_selector/core/actual_kernels/eltwise/eltwise_kernel_base.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<eltwise> : public typed_program_node_base<eltwise> {
    using parent = typed_program_node_base<eltwise>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog),
          output_qf(get_primitive()->output_quantization_factor),
          output_cf(!get_primitive()->output_calibration_factors.empty()),
          inputs_cf(!get_primitive()->inputs_calibration_factors.empty()),
          inputs_qf(!get_primitive()->input_quantization_factors.empty()) {
        support_padding_all(true);
    }

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    size_t inputs_count() const { return get_primitive()->input.size(); }
    program_node& output_calibration_factors() const { return get_dependency(inputs_count()); }
    bool output_calibration_term() const { return !get_primitive()->output_calibration_factors.empty(); }
    float get_output_qf() const { return output_qf; }

    program_node& input_calibration_factors(size_t idx = 0) const {
        size_t ocf_offset = output_calibration_term() ? 1 : 0;
        return get_dependency(inputs_count() + ocf_offset + idx);
    }
    bool inputs_calibration_term() const { return inputs_cf; }
    bool inputs_quantization_term() const { return inputs_qf; }

    std::shared_ptr<kernel_selector::fuse_params> get_fuse_params() const override {
        return std::make_shared<kernel_selector::eltwise_fuse_params>();
    }

private:
    float output_qf;
    bool output_cf;  // to know if we have calibration factors
    bool inputs_cf;
    bool inputs_qf;
};

using eltwise_node = typed_program_node<eltwise>;

template <>
class typed_primitive_inst<eltwise> : public typed_primitive_inst_base<eltwise> {
    using parent = typed_primitive_inst_base<eltwise>;
    static void check_inputs_count(eltwise_node const& node);

public:
    static layout calc_output_layout(eltwise_node const& node);
    static std::string to_string(eltwise_node const& node);

public:
    typed_primitive_inst(network_impl& network, eltwise_node const& node);

    memory_impl& output_calibration_factors_memory() const {
        return dep_memory(node.inputs_count());
    }  // because last place should be reserved for calibration factors
    bool output_calibration_factors_term() const { return node.output_calibration_term(); }

    memory_impl& input_calibration_factors_memory(size_t idx) const {
        auto inputs_offset = inputs_memory_count();
        size_t ocf_offset = node.output_calibration_term() ? 1 : 0;
        return dep_memory(inputs_offset + ocf_offset + idx);
    }

    bool inputs_calibration_factors_term() const { return node.inputs_calibration_term(); }
};

using eltwise_inst = typed_primitive_inst<eltwise>;

}  // namespace cldnn
