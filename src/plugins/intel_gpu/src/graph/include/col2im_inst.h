// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/col2im.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<col2im> : public typed_program_node_base<col2im> {
    using parent = typed_program_node_base<col2im>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<NodeFuseParams>(col2im::type_id());
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using col2im_node = typed_program_node<col2im>;

template <>
class typed_primitive_inst<col2im> : public typed_primitive_inst_base<col2im> {
    using parent = typed_primitive_inst_base<col2im>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(col2im_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(col2im_node const& node, kernel_impl_params const& impl_param);

    static std::string to_string(col2im_node const& node);

    typed_primitive_inst(network& network, col2im_node const& desc);
};

using col2im_inst = typed_primitive_inst<col2im>;
}  // namespace cldnn
