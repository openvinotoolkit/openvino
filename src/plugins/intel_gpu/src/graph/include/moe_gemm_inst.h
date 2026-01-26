// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<moe_gemm> : public typed_program_node_base<moe_gemm> {
    using parent = typed_program_node_base<moe_gemm>;
    typed_program_node(const std::shared_ptr<moe_gemm> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    program_node& input() const { return get_dependency(0); }
};

using moe_gemm_node = typed_program_node<moe_gemm>;

template <>
class typed_primitive_inst<moe_gemm> : public typed_primitive_inst_base<moe_gemm> {
    using parent = typed_primitive_inst_base<moe_gemm>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(moe_gemm_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(moe_gemm_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(moe_gemm_node const& node);

    typed_primitive_inst(network& network, moe_gemm_node const& node);
};

using moe_gemm_inst = typed_primitive_inst<moe_gemm>;
}  // namespace cldnn
