// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/depth_to_space.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<depth_to_space> : public typed_program_node_base<depth_to_space> {
    using parent = typed_program_node_base<depth_to_space>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<NodeFuseParams>(depth_to_space::type_id());
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using depth_to_space_node = typed_program_node<depth_to_space>;

template <>
class typed_primitive_inst<depth_to_space> : public typed_primitive_inst_base<depth_to_space> {
    using parent = typed_primitive_inst_base<depth_to_space>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(depth_to_space_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(depth_to_space_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(depth_to_space_node const& node);

    typed_primitive_inst(network& network, depth_to_space_node const& desc);
};

using depth_to_space_inst = typed_primitive_inst<depth_to_space>;
}  // namespace cldnn
