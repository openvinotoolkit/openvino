// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/histc.hpp"
#include "primitive_inst.h"

namespace cldnn {

using histc_node = typed_program_node<histc>;

template <>
class typed_primitive_inst<histc> : public typed_primitive_inst_base<histc> {
public:
    using parent = typed_primitive_inst_base<histc>;
    using parent::parent;

    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(histc_node const& /*node*/, const kernel_impl_params& impl_param) {
        auto primitive = impl_param.typed_desc<histc>();
        return {layout{ShapeType{primitive->bins}, *primitive->output_data_types[0], format::bfyx}};
    }

    static layout calc_output_layout(const histc_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const histc_node& node);
};

using histc_inst = typed_primitive_inst<histc>;

}  // namespace cldnn
