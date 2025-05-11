// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/scatter_update.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<scatter_update> : public typed_program_node_base<scatter_update> {
private:
    using parent = typed_program_node_base<scatter_update>;

public:
    using parent::parent;
    program_node& input(std::size_t i = 0) const { return get_dependency(i); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using scatter_update_node = typed_program_node<scatter_update>;

template <>
class typed_primitive_inst<scatter_update> : public typed_primitive_inst_base<scatter_update> {
    using parent = typed_primitive_inst_base<scatter_update>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(scatter_update_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static layout calc_output_layout(scatter_update_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(scatter_update_node const& node);

    typed_primitive_inst(network& network, scatter_update_node const& desc);
    void update_output_memory() override;

private:
    void on_execute() override;
};

using scatter_update_inst = typed_primitive_inst<scatter_update>;
}  // namespace cldnn
