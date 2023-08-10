// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "assign_inst.h"
#include "intel_gpu/primitives/read_value.hpp"
#include "primitive_inst.h"

namespace cldnn {

using read_value_node = typed_program_node<read_value>;

template<>
class typed_primitive_inst<read_value> : public typed_primitive_inst_base<read_value>, public memory_state::variable {
    using parent = typed_primitive_inst_base<read_value>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(read_value_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }

    static layout calc_output_layout(const read_value_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const read_value_node& node);

    typed_primitive_inst(network& network, const read_value_node& desc);
    typed_primitive_inst(network& network) : parent(network), memory_state::variable("") {}

    void save(cldnn::BinaryOutputBuffer& ob) const override;
    void load(cldnn::BinaryInputBuffer& ib) override;
};

using read_value_inst = typed_primitive_inst<read_value>;

} // namespace cldnn
