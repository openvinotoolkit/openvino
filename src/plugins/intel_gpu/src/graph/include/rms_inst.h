// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/rms.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<rms> : public typed_program_node_base<rms> {
    using parent = typed_program_node_base<rms>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using rms_node = typed_program_node<rms>;

template <>
class typed_primitive_inst<rms> : public typed_primitive_inst_base<rms> {
    using parent = typed_primitive_inst_base<rms>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(rms_node const& /*node*/, const kernel_impl_params& impl_params) {
        return forward_input0_shape<ShapeType>(impl_params);
    }
    static layout calc_output_layout(rms_node const& node, kernel_impl_params const& impl_params);
    static std::string to_string(rms_node const& node);

    typed_primitive_inst(network& network, rms_node const& node);
};

using rms_inst = typed_primitive_inst<rms>;

}  // namespace cldnn
