// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/rnn.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<lstm_seq> : public typed_program_node_base<lstm_seq> {
    using parent = typed_program_node_base<lstm_seq>;

public:
    using parent::parent;
    ov::op::RecurrentSequenceDirection direction() const { return get_primitive()->direction; }
};

using lstm_seq_node = typed_program_node<lstm_seq>;

template <>
class typed_primitive_inst<lstm_seq> : public typed_primitive_inst_base<lstm_seq> {
    using parent = typed_primitive_inst_base<lstm_seq>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(lstm_seq_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(lstm_seq_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(lstm_seq_node const& node);

public:
    typed_primitive_inst(network& network, lstm_seq_node const& node);
};

using lstm_seq_inst = typed_primitive_inst<lstm_seq>;
}  // namespace cldnn
