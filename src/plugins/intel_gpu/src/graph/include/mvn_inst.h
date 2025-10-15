// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/mvn.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<mvn> : public typed_program_node_base<mvn> {
    using parent = typed_program_node_base<mvn>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using mvn_node = typed_program_node<mvn>;

template <>
class typed_primitive_inst<mvn> : public typed_primitive_inst_base<mvn> {
    using parent = typed_primitive_inst_base<mvn>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(mvn_node const& /*node*/, const kernel_impl_params& impl_param)  {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static layout calc_output_layout(mvn_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(mvn_node const& node);

    typed_primitive_inst(network& network, mvn_node const& node);
};

using mvn_inst = typed_primitive_inst<mvn>;

}  // namespace cldnn
