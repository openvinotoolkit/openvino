// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/depth_to_space.hpp"
#include "primitive_inst.h"
#include "kernel_selector/core/actual_kernels/depth_to_space/depth_to_space_kernel_base.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<depth_to_space> : public typed_program_node_base<depth_to_space> {
    using parent = typed_program_node_base<depth_to_space>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::shared_ptr<kernel_selector::fuse_params> get_fuse_params() const override {
        return std::make_shared<kernel_selector::depth_to_space_fuse_params>();
    }
};

using depth_to_space_node = typed_program_node<depth_to_space>;

template <>
class typed_primitive_inst<depth_to_space> : public typed_primitive_inst_base<depth_to_space> {
    using parent = typed_primitive_inst_base<depth_to_space>;

public:
    static layout calc_output_layout(depth_to_space_node const& node);
    static std::string to_string(depth_to_space_node const& node);

public:
    typed_primitive_inst(network& network, depth_to_space_node const& desc);
};

using depth_to_space_inst = typed_primitive_inst<depth_to_space>;
}  // namespace cldnn
