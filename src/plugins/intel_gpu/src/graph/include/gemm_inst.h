// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gemm.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<gemm> : public typed_program_node_base<gemm> {
    using parent = typed_program_node_base<gemm>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using gemm_node = typed_program_node<gemm>;

template <>
class typed_primitive_inst<gemm> : public typed_primitive_inst_base<gemm> {
    using parent = typed_primitive_inst_base<gemm>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(gemm_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(gemm_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(gemm_node const& node);

    static std::vector<layout> transform_input_layouts(const std::shared_ptr<const gemm> primitive,
                                                       const std::vector<layout>& input_layouts);
    static layout transform_output_layout(const std::shared_ptr<const gemm> primitive, const std::vector<layout>& input_layouts, const layout& output_layout);

    typed_primitive_inst(network& network, gemm_node const& node);
};

using gemm_inst = typed_primitive_inst<gemm>;

}  // namespace cldnn
