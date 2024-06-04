// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/scatter_elements_update.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

using scatter_elements_update_node = typed_program_node<scatter_elements_update>;

template <>
class typed_primitive_inst<scatter_elements_update> : public typed_primitive_inst_base<scatter_elements_update> {
    using parent = typed_primitive_inst_base<scatter_elements_update>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(scatter_elements_update_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }

    static layout calc_output_layout(scatter_elements_update_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(scatter_elements_update_node const& node);

    typed_primitive_inst(network& network, scatter_elements_update_node const& desc);
    void update_output_memory() override;

private:
    void on_execute() override;
};

using scatter_elements_update_inst = typed_primitive_inst<scatter_elements_update>;
}  // namespace cldnn
