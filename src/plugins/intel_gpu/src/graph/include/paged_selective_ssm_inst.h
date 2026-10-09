// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/paged_selective_ssm.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<paged_selective_ssm> : public typed_program_node_base<paged_selective_ssm> {
    using parent = typed_program_node_base<paged_selective_ssm>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};
using paged_selective_ssm_node = typed_program_node<paged_selective_ssm>;

template <>
class typed_primitive_inst<paged_selective_ssm> : public typed_primitive_inst_base<paged_selective_ssm> {
    using parent = typed_primitive_inst_base<paged_selective_ssm>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const paged_selective_ssm_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const paged_selective_ssm_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const paged_selective_ssm_node& node);
    typed_primitive_inst(network& network, const paged_selective_ssm_node& node);
};

using paged_selective_ssm_inst = typed_primitive_inst<paged_selective_ssm>;

}  // namespace cldnn
