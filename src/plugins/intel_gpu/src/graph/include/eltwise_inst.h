// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/eltwise.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

class EltwiseFuseParams : public NodeFuseParams {
public:
    EltwiseFuseParams(std::shared_ptr<eltwise> desc) : NodeFuseParams(eltwise::type_id()), _desc(std::move(desc)) {}
    size_t ops_count() const override { return 1; }

    std::shared_ptr<eltwise> _desc;
};

template <>
struct typed_program_node<eltwise> : public typed_program_node_base<eltwise> {
    using parent = typed_program_node_base<eltwise>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog) {
        support_padding_all(true);
    }

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }

    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<EltwiseFuseParams>(typed_desc());
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using eltwise_node = typed_program_node<eltwise>;

template <>
class typed_primitive_inst<eltwise> : public typed_primitive_inst_base<eltwise> {
    using parent = typed_primitive_inst_base<eltwise>;
    using parent::parent;
    static void check_inputs_count(eltwise_node const& node);

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(eltwise_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(eltwise_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(eltwise_node const& node);
    typed_primitive_inst(network& network, eltwise_node const& node);
};

using eltwise_inst = typed_primitive_inst<eltwise>;

}  // namespace cldnn
