// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/resample.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {
template <>
struct typed_program_node<resample> : public typed_program_node_base<resample> {
    using parent = typed_program_node_base<resample>;
    typed_program_node(const std::shared_ptr<resample> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {1, 2}; }
};

using resample_node = typed_program_node<resample>;

template <>
class typed_primitive_inst<resample> : public typed_primitive_inst_base<resample> {
    using parent = typed_primitive_inst_base<resample>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(resample_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(resample_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(resample_node const& node);

public:
    typed_primitive_inst(network& network, resample_node const& node);
};

using resample_inst = typed_primitive_inst<resample>;

}  // namespace cldnn
