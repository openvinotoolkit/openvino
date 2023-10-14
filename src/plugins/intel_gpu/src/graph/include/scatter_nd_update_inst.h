// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/scatter_nd_update.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

using scatter_nd_update_node = typed_program_node<scatter_nd_update>;

template <>
class typed_primitive_inst<scatter_nd_update> : public typed_primitive_inst_base<scatter_nd_update> {
    using parent = typed_primitive_inst_base<scatter_nd_update>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(scatter_nd_update_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(scatter_nd_update_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(scatter_nd_update_node const& node);

public:
    typed_primitive_inst(network& network, scatter_nd_update_node const& desc);
    void update_output_memory() override;

private:
    void on_execute() override;
    void reuse_input();
};

using scatter_nd_update_inst = typed_primitive_inst<scatter_nd_update>;
}  // namespace cldnn
