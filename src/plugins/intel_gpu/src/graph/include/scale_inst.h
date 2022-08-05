// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/scale.hpp"
#include "primitive_inst.h"
#include "kernel_selector/core/actual_kernels/eltwise/eltwise_kernel_base.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<scale> : public typed_program_node_base<scale> {
private:
    using parent = typed_program_node_base<scale>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<scale> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
    program_node& input() const { return get_dependency(0); }
    program_node& scale_in() const { return get_dependency(1); }
    program_node& bias() const { return get_dependency(2); }

    bool bias_term() const { return get_primitive()->bias.length() != 0; }

    std::shared_ptr<kernel_selector::fuse_params> get_fuse_params() const override {
        return std::make_shared<kernel_selector::scale_fuse_params>();
    }
};

using scale_node = typed_program_node<scale>;

template <>
class typed_primitive_inst<scale> : public typed_primitive_inst_base<scale> {
    using parent = typed_primitive_inst_base<scale>;

public:
    static layout calc_output_layout(scale_node const& node);
    static std::string to_string(scale_node const& node);

public:
    typed_primitive_inst(network& network, scale_node const& desc);

    memory::ptr scale_memory() const { return dep_memory_ptr(1); }
    memory::ptr bias_memory() const { return dep_memory_ptr(2); }

    bool bias_term() const { return _node.as<scale>().bias_term(); }
};

using scale_inst = typed_primitive_inst<scale>;

}  // namespace cldnn
