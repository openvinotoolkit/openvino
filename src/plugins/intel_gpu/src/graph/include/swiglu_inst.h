// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/swiglu.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

class SwigluFuseParams : public NodeFuseParams {
public:
    SwigluFuseParams(std::shared_ptr<swiglu> desc) : NodeFuseParams(swiglu::type_id()), _desc(std::move(desc)) {}
    std::shared_ptr<swiglu> _desc;
};
template <>
struct typed_program_node<swiglu> : public typed_program_node_base<swiglu> {
    using parent = typed_program_node_base<swiglu>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }

    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<SwigluFuseParams>(typed_desc());
    }
};

using swiglu_node = typed_program_node<swiglu>;

template <>
class typed_primitive_inst<swiglu> : public typed_primitive_inst_base<swiglu> {
    using parent = typed_primitive_inst_base<swiglu>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(swiglu_node const& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(swiglu_node const& node, kernel_impl_params const& impl_params);
    static std::string to_string(swiglu_node const& node);

    typed_primitive_inst(network& network, swiglu_node const& node);
};

using swiglu_inst = typed_primitive_inst<swiglu>;

}  // namespace cldnn
