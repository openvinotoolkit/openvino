// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/lstm_cell.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<lstm_cell> : public typed_program_node_base<lstm_cell> {
    using parent = typed_program_node_base<lstm_cell>;

public:
    using parent::parent;
};

using lstm_cell_node = typed_program_node<lstm_cell>;

template <>
class typed_primitive_inst<lstm_cell> : public typed_primitive_inst_base<lstm_cell> {
    using parent = typed_primitive_inst_base<lstm_cell>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(lstm_cell_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(lstm_cell_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(lstm_cell_node const& node);

public:
    typed_primitive_inst(network& network, lstm_cell_node const& node);
};

using lstm_cell_inst = typed_primitive_inst<lstm_cell>;
}  // namespace cldnn
