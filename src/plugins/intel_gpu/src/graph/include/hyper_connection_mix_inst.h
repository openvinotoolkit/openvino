// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

#include "intel_gpu/primitives/hyper_connection_mix.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<hyper_connection_mix> : public typed_program_node_base<hyper_connection_mix> {
    using parent = typed_program_node_base<hyper_connection_mix>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};
using hyper_connection_mix_node = typed_program_node<hyper_connection_mix>;

template <>
class typed_primitive_inst<hyper_connection_mix> : public typed_primitive_inst_base<hyper_connection_mix> {
    using parent = typed_primitive_inst_base<hyper_connection_mix>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const hyper_connection_mix_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const hyper_connection_mix_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const hyper_connection_mix_node& node);
    typed_primitive_inst(network& network, const hyper_connection_mix_node& node);
};

using hyper_connection_mix_inst = typed_primitive_inst<hyper_connection_mix>;
}  // namespace cldnn
