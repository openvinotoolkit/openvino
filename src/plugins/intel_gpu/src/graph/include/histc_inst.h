// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <vector>

#include "intel_gpu/primitives/histc.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<histc> : public typed_program_node_base<histc> {
    using parent = typed_program_node_base<histc>;

public:
    using parent::parent;

    program_node& input() const {
        return get_dependency(0);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};

using histc_node = typed_program_node<histc>;

template <>
class typed_primitive_inst<histc> : public typed_primitive_inst_base<histc> {
    using parent = typed_primitive_inst_base<histc>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const histc_node& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const histc_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const histc_node& node);

    typed_primitive_inst(network& network, const histc_node& node);
};

using histc_inst = typed_primitive_inst<histc>;

}  // namespace cldnn
