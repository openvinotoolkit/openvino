// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "intel_gpu/primitives/multinomial.hpp"
#include "primitive_inst.h"

namespace cldnn {

using multinomial_node = typed_program_node<multinomial>;

template <>
class typed_primitive_inst<multinomial> : public typed_primitive_inst_base<multinomial> {
    using parent = typed_primitive_inst_base<multinomial>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(multinomial_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(multinomial_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(multinomial_node const& node);

    typed_primitive_inst(network& network, multinomial_node const& desc);
};

using multinomial_inst = typed_primitive_inst<multinomial>;

}  // namespace cldnn
