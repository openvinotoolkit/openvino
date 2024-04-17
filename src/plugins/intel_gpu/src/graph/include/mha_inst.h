// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/mha.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<mha> : public typed_program_node_base<mha> {
    using parent = typed_program_node_base<mha>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<NodeFuseParams>(mha::type_id());
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using mha_node = typed_program_node<mha>;

template <>
class typed_primitive_inst<mha> : public typed_primitive_inst_base<mha> {
    using parent = typed_primitive_inst_base<mha>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(mha_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(mha_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(mha_node const& node);

    typed_primitive_inst(network& network, mha_node const& desc);
};

using mha_inst = typed_primitive_inst<mha>;
}  // namespace cldnn
